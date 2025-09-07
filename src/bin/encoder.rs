use av_codec::encoder::Encoder;
use av_data::frame::AudioInfo;
use av_data::frame::FrameBufferConv;
use av_data::frame::FrameBufferCopy;
use av_data::frame::MediaKind;
use av_data::packet::Packet;
use av_data::rational::Rational64;
use av_data::timeinfo::TimeInfo;
use av_format::demuxer::Context as DemuxerCtx;
use av_format::muxer::Muxer;
use av_format::muxer::Writer;
use matroska::elements::SimpleBlock;
use std::any::Any;
use std::fs::File;
use std::io::Read;
use std::io::WriterPanicked;
use std::io::prelude::*;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::slice;
use std::sync::Arc;
use std::time::Duration;

use libopus::*;
use matroska::muxer::{self, MkvMuxer};

macro_rules! write_bytes_be {
  ($buf:ident, $n:ident) => {
    let bytes = $n.to_be_bytes();
    $buf[..bytes.len()].copy_from_slice(&bytes);
  };
}

/// Writes 4 unsigned bytes in a big-endian order at the start of a buffer.
#[inline]
pub fn put_u32b(buf: &mut [u8], n: u32) {
  write_bytes_be!(buf, n);
}

/// Converts an `i32` into 4 unsigned bytes and
/// writes them in a big-endian order at the start of a buffer.
#[inline]
pub fn put_i32b(buf: &mut [u8], n: i32) {
  put_u32b(buf, n as u32);
}

// trait Encode {
//   fn get_encoder(&self) -> Option<Encoder>;
// }

// #[derive(Debug)]
// struct EncodingOpts {
//   /// Input file
//   // #[structopt(parse(from_os_str))]
//   input: PathBuf,
//   /// Output file
//   // #[structopt(parse(from_os_str))]
//   output: PathBuf,
//   /// Sampling rate, in Hz
//   // #[structopt(default_value = "48000")]
//   sampling_rate: usize,
//   /// Channels, either 1 or 2
//   // #[structopt(default_value = "1")]
//   channels: usize,
//   /// Bitrate
//   // #[structopt(default_value = "16000")]
//   bits_per_second: i32,
//   /// Number of seconds to encode
//   // #[structopt(default_value = "10")]
//   seconds: usize,
// }

// impl Encode for EncodingOpts {
//   fn get_encoder(&self) -> Option<Encoder> {
//     if self.channels > 2 {
//       unimplemented!("Multichannel support")
//     }

//     let coupled_streams = if self.channels > 1 { 1 } else { 0 };

//     Encoder::create(
//       self.sampling_rate,
//       self.channels,
//       1,
//       coupled_streams,
//       &[0u8, 1u8],
//       Application::Audio,
//     )
//     .ok()
//     .map(|mut enc| {
//       enc.set_option(OPUS_SET_BITRATE_REQUEST, self.bits_per_second).unwrap();
//       enc
//         .set_option(OPUS_SET_BANDWIDTH_REQUEST, OPUS_BANDWIDTH_WIDEBAND)
//         .unwrap();
//       enc.set_option(OPUS_SET_COMPLEXITY_REQUEST, 10).unwrap();
//       enc.set_option(OPUS_SET_VBR_REQUEST, 0).unwrap();
//       enc.set_option(OPUS_SET_VBR_CONSTRAINT_REQUEST, 0).unwrap();
//       enc.set_option(OPUS_SET_PACKET_LOSS_PERC_REQUEST, 0).unwrap();
//       enc
//     })
//   }
// }
use std::io::{Seek, SeekFrom};

fn read_wav_header(
  file: &mut File,
) -> Result<(u32, u16, u32, u32), Box<dyn std::error::Error>> {
  let mut header = [0u8; 12];
  file.read_exact(&mut header)?;

  // Check RIFF signature
  if &header[0..4] != b"RIFF" {
    return Err("Not a valid RIFF file".into());
  }

  // Check WAVE signature
  if &header[8..12] != b"WAVE" {
    return Err("Not a valid WAVE file".into());
  }

  let mut channels = 0u16;
  let mut sample_rate = 0u32;
  let mut bits_per_sample = 0u16;
  let mut data_size = 0u32;
  let mut fmt_found = false;
  let mut data_found = false;

  // Read chunks until we find both fmt and data chunks
  while !fmt_found || !data_found {
    let mut chunk_header = [0u8; 8];
    if file.read_exact(&mut chunk_header).is_err() {
      return Err("Unexpected end of file while reading chunks".into());
    }

    let chunk_id = &chunk_header[0..4];
    let chunk_size = u32::from_le_bytes([
      chunk_header[4],
      chunk_header[5],
      chunk_header[6],
      chunk_header[7],
    ]);

    match chunk_id {
      b"fmt " => {
        if chunk_size < 16 {
          return Err("Invalid fmt chunk size".into());
        }

        let mut fmt_data = vec![0u8; chunk_size as usize];
        file.read_exact(&mut fmt_data)?;

        // Parse format data
        let audio_format = u16::from_le_bytes([fmt_data[0], fmt_data[1]]);
        if audio_format != 1 {
          return Err("Only PCM format is supported".into());
        }

        channels = u16::from_le_bytes([fmt_data[2], fmt_data[3]]);
        sample_rate = u32::from_le_bytes([
          fmt_data[4],
          fmt_data[5],
          fmt_data[6],
          fmt_data[7],
        ]);
        bits_per_sample = u16::from_le_bytes([fmt_data[14], fmt_data[15]]);

        fmt_found = true;
      }
      b"data" => {
        data_size = chunk_size;
        data_found = true;
        // Don't read the data, just note where it starts
      }
      _ => {
        // Skip unknown chunks
        file.seek(SeekFrom::Current(chunk_size as i64))?;
      }
    }
  }

  if !fmt_found {
    return Err("fmt chunk not found".into());
  }
  if !data_found {
    return Err("data chunk not found".into());
  }

  println!(
    "WAV file info: {} channels, {} Hz, {} bits, {} bytes of audio data",
    channels, sample_rate, bits_per_sample, data_size
  );

  Ok((sample_rate, channels, bits_per_sample as u32, data_size))
}

fn main() {
  // let enc_opt = EncodingOpts {
  //   input: PathBuf::from("test.wav"),
  //   output: PathBuf::from("output.mkv"),
  //   sampling_rate: 48_000,
  //   channels: 1,
  //   bits_per_second: 16_000,
  //   seconds: 5,
  // };

  let mut in_f = File::open(PathBuf::from("test.wav")).unwrap();
  // let mut out_f = File::create(enc_opt.output).unwrap();
  let (wav_sample_rate, wav_channels, wav_bits_per_sample, audio_data_size) =
    read_wav_header(&mut in_f).unwrap();

  if wav_sample_rate != 48000 {
    panic!("Expected 48kHz sample rate, got {}", wav_sample_rate);
  }
  if wav_channels != 1 {
    panic!("Expected mono audio, got {} channels", wav_channels);
  }
  if wav_bits_per_sample != 16 {
    panic!("Expected 16-bit audio, got {} bits", wav_bits_per_sample);
  }

  use av_codec::encoder::{Descriptor, Encoder};

  let channels: usize = wav_channels as usize;

  let mut enc = OPUS_DESCR.create();
  enc
    .set_option("channels", av_data::value::Value::U64(channels as u64))
    .unwrap();
  enc.configure().unwrap();

  let mut mux = av_format::muxer::Context::new(
    MkvMuxer::matroska(),
    Writer::new(Vec::new()),
  );

  let codec_params = enc.get_params().unwrap();

  let mkv_timebase = av_data::rational::Rational64::new(1, 1000);

  let sample_rate = wav_sample_rate as i64;
  let sample_timebase = av_data::rational::Rational64::new(1, sample_rate);

  let mut abs_pts_samples: i64 = 0; // PTS in samples
  let mut cluster_time_ms: i64 = 0; // cluster base time in milliseconds
  let mut cluster_started = false;

  let bytes_per_sample = (wav_bits_per_sample / 8) as u32;
  let total_samples =
    audio_data_size / (bytes_per_sample * wav_channels as u32);
  let duration_seconds = total_samples as f64 / wav_sample_rate as f64;

  let stream_info = av_format::stream::Stream {
    id: 0,
    index: 0,
    params: codec_params.clone(),
    start: None,
    duration: Some(duration_seconds as u64),
    timebase: sample_timebase,
    user_private: None,
  };

  mux
    .set_global_info(av_format::common::GlobalInfo {
      duration: Some(duration_seconds as u64),
      timebase: Some(sample_timebase),
      streams: vec![stream_info],
    })
    .unwrap();

  mux.configure().unwrap();
  mux.write_header().unwrap();

  println!(
    "Total samples: {}, Duration: {:.2} seconds",
    total_samples, duration_seconds
  );

  // let frame_size = 960;
  // let bytes_per_frame = frame_size * channels * bytes_per_sample as usize;
  // let mut buf = vec![0u8; bytes_per_frame];
  // let mut bytes_read_total = 0u32;

  // read raw PCM (dummy: replace with proper WAV reader)
  // let frame_size = 960;
  // let mut buf = vec![0u8; frame_size * channels as usize * 2];
  let frame_size = 960;
  let bytes_per_frame = frame_size * channels * bytes_per_sample as usize;
  let mut buf = vec![0u8; bytes_per_frame];
  let mut bytes_read_total = 0u32;

  while bytes_read_total < audio_data_size {
    let bytes_remaining = audio_data_size - bytes_read_total;
    let bytes_to_read = bytes_per_frame.min(bytes_remaining as usize);

    // Resize buffer if this is the last partial frame
    if bytes_to_read < bytes_per_frame {
      buf.resize(bytes_to_read, 0);
    }

    while let Ok(_) = in_f.read_exact(&mut buf) {
      bytes_read_total += bytes_to_read as u32;

      // Convert bytes to samples
      let samples_in_this_frame =
        bytes_to_read / (channels * bytes_per_sample as usize);
      let samples: &[i16] = unsafe {
        slice::from_raw_parts(
          buf.as_ptr() as *const i16,
          samples_in_this_frame * channels,
        )
      };

      // If this is a partial frame, pad with zeros
      let mut padded_samples = vec![0i16; frame_size * channels];
      let copy_len = samples.len().min(padded_samples.len());
      padded_samples[..copy_len].copy_from_slice(&samples[..copy_len]);

      // Create frame
      let frame_kind = match codec_params.clone().kind.unwrap() {
        av_data::params::MediaKind::Audio(a) => {
          av_data::frame::MediaKind::Audio(AudioInfo {
            samples: frame_size,
            sample_rate: a.rate,
            map: a.map.unwrap(),
            format: a.format.unwrap(),
            block_len: Some(frame_size),
          })
        }
        av_data::params::MediaKind::Video(_) => {
          unimplemented!()
        }
      };

      let mut frame =
        av_data::frame::Frame::new_default_frame(frame_kind, None);

      use av_data::frame::FrameBufferConv;

      FrameBufferConv::<i16>::as_mut_slice(frame.buf.as_mut(), 0)
        .unwrap()
        .copy_from_slice(&padded_samples);

      frame.t.pts = Some(abs_pts_samples);
      frame.t.dts = Some(abs_pts_samples);
      frame.t.duration = Some(frame_size as u64);
      frame.t.timebase = Some(sample_timebase);

      let arc_frame = Arc::new(frame);

      enc.send_frame(&arc_frame).unwrap();

      while let Ok(pkt) = enc.receive_packet() {
        // Convert packet PTS from samples to milliseconds
        let pts_samples = pkt.t.pts.unwrap_or(abs_pts_samples);
        let pts_ms = (pts_samples * 1000) / sample_rate;

        // Initialize cluster on first packet
        if !cluster_started {
          cluster_time_ms = pts_ms;
          cluster_started = true;
        }

        // Calculate relative timestamp for this cluster
        let mut rel_ts_ms = pts_ms - cluster_time_ms;

        // Start new cluster if relative timestamp exceeds i16 range
        if rel_ts_ms > i16::MAX as i64 || rel_ts_ms < i16::MIN as i64 {
          cluster_time_ms = pts_ms;
          rel_ts_ms = 0;
        }

        // Create packet for muxer
        let mut mux_pkt = pkt.clone();
        mux_pkt.stream_index = 0;
        mux_pkt.t.pts = Some(pts_ms);
        mux_pkt.t.dts = Some(pts_ms);
        mux_pkt.t.timebase = Some(mkv_timebase);

        if let Some(duration_samples) = pkt.t.duration {
          let duration_ms = ((duration_samples * 1000)
            + (sample_rate as u64 / 2))
            / sample_rate as u64;
          let duration_ms = duration_ms.max(1);
          mux_pkt.t.duration = Some(duration_ms);
        }

        println!(
          "Writing packet: PTS={}ms, samples processed: {}/{}",
          pts_ms,
          bytes_read_total / bytes_per_sample / wav_channels as u32,
          total_samples
        );

        mux.write_packet(Arc::new(mux_pkt)).unwrap();
      }

      abs_pts_samples += frame_size as i64;

      // Break if we've processed all the audio data
      if bytes_read_total >= audio_data_size {
        break;
      }
    }
  }

  // while let Ok(_) = in_f.read_exact(&mut buf) {
  //   // let samples: &[i16] =
  //   //   unsafe { slice::from_raw_parts(buf.as_ptr() as *const i16, frame_size) };

  //   let samples: &[i16] = unsafe {
  //     slice::from_raw_parts(
  //       buf.as_ptr() as *const i16,
  //       frame_size * channels as usize,
  //     )
  //   };
  //   // wrap PCM into ArcFrame

  //   let frame_kind = match codec_params.clone().kind.unwrap() {
  //     av_data::params::MediaKind::Audio(a) => {
  //       av_data::frame::MediaKind::Audio(AudioInfo {
  //         samples: frame_size,
  //         sample_rate: a.rate,
  //         map: a.map.unwrap(),
  //         format: a.format.unwrap(),
  //         // block_len: Some(frame_size * channels),
  //         block_len: Some(frame_size),
  //       })
  //     }
  //     av_data::params::MediaKind::Video(_) => {
  //       unimplemented!()
  //     }
  //   };

  //   let mut frame = av_data::frame::Frame::new_default_frame(frame_kind, None);

  //   use av_data::frame::FrameBufferConv;

  //   FrameBufferConv::<i16>::as_mut_slice(frame.buf.as_mut(), 0)
  //     .unwrap()
  //     .copy_from_slice(samples);

  //   frame.t.pts = Some(abs_pts_samples);
  //   frame.t.dts = Some(abs_pts_samples);
  //   frame.t.duration = Some(frame_size as u64);
  //   frame.t.timebase = Some(sample_timebase);

  //   let arc_frame = Arc::new(frame);

  //   enc.send_frame(&arc_frame).unwrap();

  //   while let Ok(pkt) = enc.receive_packet() {
  //     write_packet_to_mux(
  //       &mut mux,
  //       pkt,
  //       &mut cluster_time_ms,
  //       &mut cluster_started,
  //       sample_rate as i64,
  //     )
  //     .unwrap();
  //   }

  //   abs_pts_samples += frame_size as i64;
  // }

  // enc.flush().unwrap();
  // while let Ok(pkt) = enc.receive_packet() {
  //   write_packet_to_mux(
  //     &mut mux,
  //     pkt,
  //     &mut cluster_time_ms,
  //     &mut cluster_started,
  //     sample_rate as i64,
  //   )
  //   .unwrap();
  // }

  mux.write_trailer().unwrap();

  // Get the output data and write to file
  let output_data = mux.writer().as_ref().0.clone();
  println!("Generated MKV data size: {} bytes", output_data.len());

  if output_data.is_empty() {
    println!("Error: No data generated!");
  } else {
    std::fs::write("output.mkv", &output_data).unwrap();
    println!("Written {} bytes to output.mkv", output_data.len());

    // Check file exists and size
    let metadata = std::fs::metadata("output.mkv").unwrap();
    println!("File size on disk: {} bytes", metadata.len());
  }

  // while let Ok(pkt) = enc.receive_packet() {}

  // let codec_params = enc.get_params();

  // encoder_trait
  // OPUS_DESCR.create()

  // let mut enc = OPUS_DESCR::cre

  // let mut enc = enc_opt.get_encoder().unwrap();
  // let frame_size = 2880;
  // let total_bytes =
  //   (enc_opt.channels * enc_opt.seconds * enc_opt.sampling_rate * 2) as usize;
  // let max_packet = 1500;
  // let mut processed_bytes = 0;
  // let mut buf = Vec::with_capacity(frame_size * 2);
  // let mut out_buf = Vec::with_capacity(max_packet);
  // buf.resize(frame_size * 2, 0u8);
  // out_buf.resize(max_packet, 0u8);

  // let mut m = av_format::muxer::Context::new(
  //   MkvMuxer::matroska(),
  //   Writer::new(Vec::new()),
  // );

  // m.configure().unwrap();
  // m.write_header().unwrap();

  // while processed_bytes < total_bytes {
  //   in_f.read_exact(&mut buf).unwrap();

  //   let samples: &[i16] =
  //     unsafe { slice::from_raw_parts(buf.as_ptr() as *const i16, frame_size) };

  //   processed_bytes += frame_size * 2;

  //   if let Ok(ret) = enc.encode(samples, &mut out_buf) {
  //     let mut b = [0u8; 4];
  //     // Write the packet size
  //     put_i32b(&mut b, ret as i32);
  //     out_f.write_all(&b).unwrap();

  //     // Write the encoder ec final state
  //     let val = enc.get_option(OPUS_GET_FINAL_RANGE_REQUEST).unwrap();
  //     put_i32b(&mut b, val);
  //     out_f.write_all(&b).unwrap();

  //     // Write the actual packet
  //     // out_f.write_all(&out_buf[..ret]).unwrap();

  //     let data = out_buf[..ret].to_vec();
  //     let mut pkt = Packet::new();
  //     pkt.data = data;

  //     let samples_encoded = processed_bytes / 2 / enc_opt.channels;
  //     pkt.t.pts = Some(samples_encoded as i64);
  //     pkt.t.dts = pkt.t.pts;
  //     pkt.t.duration = Some(frame_size as u64);
  //     pkt.stream_index = 0;

  //     m.write_packet(Arc::new(pkt)).unwrap();
  //   } else {
  //     panic!("Cannot encode");
  //   }
  // }

  // m.write_trailer().unwrap();
  // out_f.write_all(m.writer().as_ref().0).unwrap();
}
