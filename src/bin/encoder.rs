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

  use av_codec::encoder::{Descriptor, Encoder};

  let channels: usize = 1;
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

  let sample_rate = 48000;
  let sample_timebase = av_data::rational::Rational64::new(1, sample_rate);

  let stream_info = av_format::stream::Stream {
    id: 0,
    index: 0,
    params: codec_params.clone(),
    start: None,
    duration: None,
    timebase: sample_timebase,
    user_private: None,
  };

  mux
    .set_global_info(av_format::common::GlobalInfo {
      duration: None,
      timebase: Some(sample_timebase),
      streams: vec![stream_info],
    })
    .unwrap();

  mux.configure().unwrap();
  mux.write_header().unwrap();

  let mut abs_pts_samples: i64 = 0; // PTS in samples
  let mut cluster_time_ms: i64 = 0; // cluster base time in milliseconds
  let mut cluster_started = false;

  // read raw PCM (dummy: replace with proper WAV reader)
  let frame_size = 960;
  let mut buf = vec![0u8; frame_size * channels as usize * 2];
  while let Ok(_) = in_f.read_exact(&mut buf) {
    // let samples: &[i16] =
    //   unsafe { slice::from_raw_parts(buf.as_ptr() as *const i16, frame_size) };

    let samples: &[i16] = unsafe {
      slice::from_raw_parts(
        buf.as_ptr() as *const i16,
        frame_size * channels as usize,
      )
    };
    // wrap PCM into ArcFrame

    let frame_kind = match codec_params.clone().kind.unwrap() {
      av_data::params::MediaKind::Audio(a) => {
        av_data::frame::MediaKind::Audio(AudioInfo {
          samples: frame_size,
          sample_rate: a.rate,
          map: a.map.unwrap(),
          format: a.format.unwrap(),
          // block_len: Some(frame_size * channels),
          block_len: Some(frame_size),
        })
      }
      av_data::params::MediaKind::Video(_) => {
        unimplemented!()
      }
    };

    let mut frame = av_data::frame::Frame::new_default_frame(frame_kind, None);

    use av_data::frame::FrameBufferConv;

    FrameBufferConv::<i16>::as_mut_slice(frame.buf.as_mut(), 0)
      .unwrap()
      .copy_from_slice(samples);

    frame.t.pts = Some(abs_pts_samples);
    frame.t.dts = Some(abs_pts_samples);
    frame.t.duration = Some(frame_size as u64);
    frame.t.timebase = Some(sample_timebase);

    let arc_frame = Arc::new(frame);

    enc.send_frame(&arc_frame).unwrap();

    while let Ok(pkt) = enc.receive_packet() {
      // Convert packet PTS from samples to milliseconds
      // let pts_samples = pkt.t.pts.unwrap_or(abs_pts_samples);
      // let pts_ms = (pts_samples * 1000) / sample_rate as i64;

      // // Initialize cluster on first packet
      // if !cluster_started {
      //   cluster_time_ms = pts_ms;
      //   cluster_started = true;
      // }

      // // Calculate relative timestamp for this cluster
      // let mut rel_ts_ms = pts_ms - cluster_time_ms;

      // // Start new cluster if relative timestamp exceeds i16 range
      // // MKV uses signed 16-bit relative timestamps
      // if rel_ts_ms > i16::MAX as i64 || rel_ts_ms < i16::MIN as i64 {
      //   cluster_time_ms = pts_ms;
      //   rel_ts_ms = 0;
      // }

      // // Create packet for muxer with millisecond timebase
      // let mut mux_pkt = pkt.clone();
      // mux_pkt.stream_index = 0;

      let pts_samples = pkt.t.pts.unwrap_or(0);
      let pts_ms = (pts_samples * 1000) / sample_rate;

      // Initialize cluster on first packet
      if !*cluster_started {
        *cluster_time_ms = pts_ms;
        *cluster_started = true;
      }

      // Calculate relative timestamp for this cluster
      let mut rel_ts_ms = pts_ms - *cluster_time_ms;

      // Start new cluster if relative timestamp exceeds i16 range
      // MKV uses signed 16-bit relative timestamps
      if rel_ts_ms > i16::MAX as i64 || rel_ts_ms < i16::MIN as i64 {
        *cluster_time_ms = pts_ms;
        rel_ts_ms = 0;
      }

      // Create packet for muxer with consistent timebase
      let mut mux_pkt = pkt.clone();
      mux_pkt.stream_index = 0;

      // Set timestamps in milliseconds with proper timebase
      mux_pkt.t.pts = Some(pts_ms);
      mux_pkt.t.dts = Some(pts_ms);
      mux_pkt.t.timebase = Some(mkv_timebase);

      // Calculate duration in milliseconds
      if let Some(duration_samples) = pkt.t.duration {
        let duration_ms = ((duration_samples * 1000)
          + (sample_rate as u64 / 2))
          / sample_rate as u64;
        let duration_ms = duration_ms.max(1);
        mux_pkt.t.duration = Some(duration_ms);
      }

      println!("Writing packet: PTS={}ms, rel_ts={}ms", pts_ms, rel_ts_ms);

      // Set absolute PTS/DTS in milliseconds for the muxer
      // mux_pkt.t.pts = Some(pts_ms);
      // mux_pkt.t.dts = Some(pts_ms);
      // mux_pkt.t.timebase = Some(mkv_timebase);

      // Duration in milliseconds
      // let duration_samples = pkt.t.duration.unwrap_or(frame_size as u64);
      // let duration_ms = (duration_samples * 1000) / sample_rate as u64;
      // mux_pkt.t.duration = Some(duration_ms);
      // let duration_samples = pkt.t.duration.unwrap_or(frame_size as u64);
      // let duration_ms = ((duration_samples * 1000) + (sample_rate as u64 / 2))
      //   / sample_rate as u64;
      // let duration_ms = duration_ms.max(1);
      // mux_pkt.t.duration = Some(duration_ms);

      // println!(
      //   "Writing packet: PTS={}ms, rel_ts={}ms, duration={}ms",
      //   pts_ms, rel_ts_ms
      // );

      // let pts = pkt.t.pts.unwrap();
      // let mut rel_ts = (pts - cluster_time) as i32;

      // // if rel_ts would overflow i16, start a new cluster
      // if rel_ts > i16::MAX as i32 || rel_ts < i16::MIN as i32 {
      //   cluster_time = pts;
      //   rel_ts = 0;
      // }

      // // clamp to i16
      // let rel_ts_i16 = rel_ts.clamp(i16::MIN as i32, i16::MAX as i32) as i16;

      // // update packet timestamps (optional, for muxer)
      // let mut pkt = pkt.clone();
      // pkt.stream_index = 0; // if mono audio, first track

      // pkt.t.pts = Some(cluster_time + rel_ts_i16 as i64);
      // pkt.t.dts = Some(cluster_time + rel_ts_i16 as i64);
      println!("{:?}", mux_pkt);

      mux.write_packet(Arc::new(mux_pkt)).unwrap();
    }

    abs_pts_samples += frame_size as i64;
  }

  enc.flush().unwrap();
  while let Ok(pkt) = enc.receive_packet() {
    let pts_samples = pkt.t.pts.unwrap_or(abs_pts_samples);
    let pts_ms = (pts_samples * 1000) / sample_rate as i64;

    let mut mux_pkt = pkt.clone();
    mux_pkt.stream_index = 0;
    mux_pkt.t.pts = Some(pts_ms);
    mux_pkt.t.dts = Some(pts_ms);
    mux_pkt.t.timebase = Some(mkv_timebase);

    let duration_samples = pkt.t.duration.unwrap_or(frame_size as u64);
    let duration_ms = ((duration_samples * 1000) + (sample_rate as u64 / 2))
      / sample_rate as u64;
    let duration_ms = duration_ms.max(1);
    mux_pkt.t.duration = Some(duration_ms);

    println!(
      "Flushing packet: PTS={}ms, duration={}ms (samples={})",
      pts_ms, duration_ms, duration_samples
    );
    mux.write_packet(Arc::new(mux_pkt)).unwrap();
  }

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
