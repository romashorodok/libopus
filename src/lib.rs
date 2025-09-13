mod libopus;

// use crate::libopus::*;
use std::ffi::CStr;
use std::fmt;
use std::ptr;
use std::str::FromStr;

pub use libopus::*;

#[repr(i32)]
#[derive(Copy, Clone, Debug)]
pub enum ErrorCode {
  BadArg = OPUS_BAD_ARG,
  BufferTooSmall = OPUS_BUFFER_TOO_SMALL,
  InternalError = OPUS_INTERNAL_ERROR,
  InvalidPacket = OPUS_INVALID_PACKET,
  Unimplemented = OPUS_UNIMPLEMENTED,
  InvalidState = OPUS_INVALID_STATE,
  AllocFail = OPUS_ALLOC_FAIL,
  Unknown = i32::MAX,
}

impl fmt::Display for ErrorCode {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    let v = *self;
    let s = unsafe { CStr::from_ptr(opus_strerror(v as i32)) };
    write!(f, "{}", s.to_string_lossy())
  }
}

impl From<i32> for ErrorCode {
  fn from(v: i32) -> Self {
    use self::ErrorCode::*;
    match v {
      OPUS_BAD_ARG => BadArg,
      OPUS_BUFFER_TOO_SMALL => BufferTooSmall,
      OPUS_INTERNAL_ERROR => InternalError,
      OPUS_INVALID_PACKET => InvalidPacket,
      OPUS_UNIMPLEMENTED => Unimplemented,
      OPUS_INVALID_STATE => InvalidState,
      OPUS_ALLOC_FAIL => AllocFail,
      _ => Unknown,
    }
  }
}

pub enum AudioBuffer<'a> {
  F32(&'a [f32]),
  I16(&'a [i16]),
}

impl<'a> From<&'a [i16]> for AudioBuffer<'a> {
  fn from(v: &'a [i16]) -> Self {
    AudioBuffer::I16(v)
  }
}

impl<'a> From<&'a [f32]> for AudioBuffer<'a> {
  fn from(v: &'a [f32]) -> Self {
    AudioBuffer::F32(v)
  }
}

pub enum AudioBufferMut<'a> {
  F32(&'a mut [f32]),
  I16(&'a mut [i16]),
}

impl<'a> From<&'a mut [f32]> for AudioBufferMut<'a> {
  fn from(v: &'a mut [f32]) -> Self {
    AudioBufferMut::F32(v)
  }
}

impl<'a> From<&'a mut [i16]> for AudioBufferMut<'a> {
  fn from(v: &'a mut [i16]) -> Self {
    AudioBufferMut::I16(v)
  }
}

pub struct Decoder {
  dec: *mut OpusMSDecoder,
  channels: usize,
}

unsafe impl Send for Decoder {}
unsafe impl Sync for Decoder {}

impl Decoder {
  pub fn create(
    sample_rate: usize, channels: usize, streams: usize,
    coupled_streams: usize, mapping: &[u8],
  ) -> Result<Decoder, ErrorCode> {
    let mut err = 0;
    let dec = unsafe {
      opus_multistream_decoder_create(
        sample_rate as i32,
        channels as i32,
        streams as i32,
        coupled_streams as i32,
        mapping.as_ptr(),
        &mut err,
      )
    };

    if err < 0 { Err(err.into()) } else { Ok(Decoder { dec, channels }) }
  }

  pub fn decode<'a, I, O>(
    &mut self, input: I, out: O, decode_fec: bool,
  ) -> Result<usize, ErrorCode>
  where
    I: Into<Option<&'a [u8]>>,
    O: Into<AudioBufferMut<'a>>,
  {
    let (data, len) =
      input.into().map_or((ptr::null(), 0), |v| (v.as_ptr(), v.len()));

    let ret = match out.into() {
      AudioBufferMut::F32(v) => unsafe {
        opus_multistream_decode_float(
          self.dec,
          data,
          len as i32,
          v.as_mut_ptr(),
          (v.len() / self.channels) as i32,
          decode_fec as i32,
        )
      },
      AudioBufferMut::I16(v) => unsafe {
        opus_multistream_decode(
          self.dec,
          data,
          len as i32,
          v.as_mut_ptr(),
          (v.len() / self.channels) as i32,
          decode_fec as i32,
        )
      },
    };

    if ret < 0 { Err(ret.into()) } else { Ok(ret as usize) }
  }

  pub fn set_option(&mut self, key: i32, val: i32) -> Result<(), ErrorCode> {
    let ret = match key {
      OPUS_SET_GAIN_REQUEST => unsafe {
        opus_multistream_decoder_ctl(self.dec, key, val)
      },
      _ => unimplemented!(),
    };

    if ret < 0 { Err(ret.into()) } else { Ok(()) }
  }

  pub fn reset(&mut self) {
    let _ =
      unsafe { opus_multistream_decoder_ctl(self.dec, OPUS_RESET_STATE) };
  }
}

impl Drop for Decoder {
  fn drop(&mut self) {
    unsafe { opus_multistream_decoder_destroy(self.dec) }
  }
}

pub struct Encoder {
  enc: *mut OpusMSEncoder,
  channels: usize,
}

unsafe impl Send for Encoder {} // TODO: Make sure it cannot be abused

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
pub enum Application {
  Voip = OPUS_APPLICATION_VOIP,
  Audio = OPUS_APPLICATION_AUDIO,
  LowDelay = OPUS_APPLICATION_RESTRICTED_LOWDELAY,
}

impl FromStr for Application {
  type Err = ();

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    use self::Application::*;
    match s {
      "voip" => Ok(Voip),
      "audio" => Ok(Audio),
      "lowdelay" => Ok(LowDelay),
      _ => Err(()),
    }
  }
}

impl Encoder {
  pub fn create(
    sample_rate: usize, channels: usize, streams: usize,
    coupled_streams: usize, mapping: &[u8], application: Application,
  ) -> Result<Encoder, ErrorCode> {
    let mut err = 0;
    let enc = unsafe {
      opus_multistream_encoder_create(
        sample_rate as i32,
        channels as i32,
        streams as i32,
        coupled_streams as i32,
        mapping.as_ptr(),
        application as i32,
        &mut err,
      )
    };

    if err < 0 { Err(err.into()) } else { Ok(Encoder { enc, channels }) }
  }

  pub fn encode<'a, I>(
    &mut self, input: I, output: &mut [u8],
  ) -> Result<usize, ErrorCode>
  where
    I: Into<AudioBuffer<'a>>,
  {
    let ret = match input.into() {
      AudioBuffer::F32(v) => unsafe {
        opus_multistream_encode_float(
          self.enc,
          v.as_ptr(),
          (v.len() / self.channels) as i32,
          output.as_mut_ptr(),
          output.len() as i32,
        )
      },
      AudioBuffer::I16(v) => unsafe {
        opus_multistream_encode(
          self.enc,
          v.as_ptr(),
          (v.len() / self.channels) as i32,
          output.as_mut_ptr(),
          output.len() as i32,
        )
      },
    };

    if ret < 0 { Err(ret.into()) } else { Ok(ret as usize) }
  }

  pub fn set_option(&mut self, key: i32, val: i32) -> Result<(), ErrorCode> {
    let ret = match key {
      OPUS_SET_APPLICATION_REQUEST
      | OPUS_SET_BITRATE_REQUEST
      | OPUS_SET_MAX_BANDWIDTH_REQUEST
      | OPUS_SET_VBR_REQUEST
      | OPUS_SET_BANDWIDTH_REQUEST
      | OPUS_SET_COMPLEXITY_REQUEST
      | OPUS_SET_INBAND_FEC_REQUEST
      | OPUS_SET_PACKET_LOSS_PERC_REQUEST
      | OPUS_SET_DTX_REQUEST
      | OPUS_SET_VBR_CONSTRAINT_REQUEST
      | OPUS_SET_FORCE_CHANNELS_REQUEST
      | OPUS_SET_SIGNAL_REQUEST
      | OPUS_SET_GAIN_REQUEST
      | OPUS_SET_LSB_DEPTH_REQUEST
      | OPUS_SET_EXPERT_FRAME_DURATION_REQUEST
      | OPUS_SET_PREDICTION_DISABLED_REQUEST
      // | OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST
      => unsafe {
        opus_multistream_encoder_ctl(self.enc, key, val)
      },
      _ => unimplemented!(),
    };

    if ret < 0 { Err(ret.into()) } else { Ok(()) }
  }

  pub fn get_option(&self, key: i32) -> Result<i32, ErrorCode> {
    let mut val: i32 = 0;
    let ret = match key {
      OPUS_GET_LOOKAHEAD_REQUEST | OPUS_GET_FINAL_RANGE_REQUEST => unsafe {
        opus_multistream_encoder_ctl(self.enc, key, &mut val as *mut i32)
      },
      _ => unimplemented!(),
    };

    if ret < 0 { Err(ret.into()) } else { Ok(val) }
  }

  pub fn reset(&mut self) {
    let _ =
      unsafe { opus_multistream_encoder_ctl(self.enc, OPUS_RESET_STATE) };
  }
}

impl Drop for Encoder {
  fn drop(&mut self) {
    unsafe { opus_multistream_encoder_destroy(self.enc) };
  }
}

pub mod encoder_trait {
  use super::Application;
  use super::Encoder as OpusEncoder;
  use crate::libopus::*;
  use av_codec::encoder::*;
  use av_codec::error::*;
  use av_data::audiosample::ChannelMap;
  use av_data::audiosample::formats::S16;
  use av_data::frame::{ArcFrame, FrameBufferConv, MediaKind};
  use av_data::packet::Packet;
  use av_data::params::CodecParams;
  use av_data::value::Value;
  use std::collections::VecDeque;

  pub struct Des {
    descr: Descr,
  }

  struct Cfg {
    channels: usize,
    streams: usize,
    coupled_streams: usize,
    mapping: Vec<u8>,
    application: Application,
    bitrate: usize,
  }

  impl Cfg {
    fn is_valid(&self) -> bool {
      self.channels > 0
        && self.streams + self.coupled_streams == self.channels
        && self.mapping.len() == self.channels
    }
  }

  pub struct Enc {
    enc: Option<OpusEncoder>,
    pending: VecDeque<Packet>,
    frame_size: usize,
    delay: usize,
    cfg: Cfg,
    flushing: bool,
  }

  impl Descriptor for Des {
    type OutputEncoder = Enc;

    fn create(&self) -> Self::OutputEncoder {
      Enc {
        enc: None,
        pending: VecDeque::new(),
        frame_size: 960,
        delay: 0,
        cfg: Cfg {
          channels: 1,
          streams: 1,
          coupled_streams: 0,
          mapping: vec![0],
          application: Application::Audio,
          bitrate: 16000,
        },
        flushing: false,
      }
      // Stereo
      // cfg: Cfg {
      //   channels: 2,
      //   streams: 1,
      //   coupled_streams: 1,
      //   mapping: vec![0, 1],
      //   application: Application::Audio,
      //   bitrate: 32000,
      // }
    }

    fn describe(&self) -> &Descr {
      &self.descr
    }
  }

  // Values copied from libopusenc.c
  // A packet may contain up to 3 frames, each of 1275 bytes max.
  // The packet header may be up to 7 bytes long.

  const MAX_HEADER_SIZE: usize = 7;
  const MAX_FRAME_SIZE: usize = 1275;
  const MAX_FRAMES: usize = 3;

  /// 80ms in samples
  const CONVERGENCE_WINDOW: usize = 3840;

  impl Encoder for Enc {
    fn configure(&mut self) -> Result<()> {
      if self.enc.is_none() {
        if self.cfg.is_valid() {
          let mut enc = OpusEncoder::create(
            48000, // TODO
            self.cfg.channels,
            self.cfg.streams,
            self.cfg.coupled_streams,
            &self.cfg.mapping,
            self.cfg.application,
          )
          .map_err(|_e| unimplemented!())?;
          enc
            .set_option(OPUS_SET_BITRATE_REQUEST, self.cfg.bitrate as i32)
            .unwrap();
          enc
            .set_option(OPUS_SET_BANDWIDTH_REQUEST, OPUS_BANDWIDTH_WIDEBAND)
            .unwrap();
          enc.set_option(OPUS_SET_COMPLEXITY_REQUEST, 10).unwrap();
          enc.set_option(OPUS_SET_VBR_REQUEST, 0).unwrap();
          enc.set_option(OPUS_SET_VBR_CONSTRAINT_REQUEST, 0).unwrap();
          enc.set_option(OPUS_SET_PACKET_LOSS_PERC_REQUEST, 0).unwrap();

          self.delay =
            enc.get_option(OPUS_GET_LOOKAHEAD_REQUEST).unwrap() as usize;
          self.enc = Some(enc);
          Ok(())
        } else {
          unimplemented!()
        }
      } else {
        unimplemented!()
      }
    }
    // TODO: support multichannel
    fn get_extradata(&self) -> Option<Vec<u8>> {
      use av_bitstream::bytewrite::*;
      if self.cfg.channels > 2 {
        unimplemented!();
      }

      let mut buf = b"OpusHead".to_vec();

      buf.resize(19, 0);

      buf[8] = 1;
      buf[9] = self.cfg.channels as u8;
      put_i16l(&mut buf[10..12], self.delay as i16);
      put_i32l(&mut buf[12..16], 48000); // TODO
      put_i16l(&mut buf[16..18], 0);
      buf[18] = 0;

      Some(buf)
    }

    fn send_frame(&mut self, frame: &ArcFrame) -> Result<()> {
      let enc = self.enc.as_mut().unwrap();
      let pending = &mut self.pending;
      if let MediaKind::Audio(ref info) = frame.kind {
        let channels = info.map.len();
        let input_size = info.samples * channels;
        let input: &[i16] = frame.buf.as_slice(0).unwrap();
        let data_size = MAX_HEADER_SIZE + MAX_FRAMES * MAX_FRAME_SIZE;
        let chunk_size = self.frame_size * channels;
        let mut buf = Vec::with_capacity(chunk_size);
        let mut pts = frame.t.pts.unwrap();

        for chunk in input[..input_size].chunks(chunk_size) {
          let len = chunk.len();
          let mut pkt = Packet::with_capacity(data_size);

          pkt.data.resize(data_size, 0); // TODO is it needed?

          let input_data = if len < chunk_size {
            buf.clear();
            buf.extend_from_slice(chunk);
            // buf.resize(chunk_size, 0);
            buf.as_slice()
            // buf.as_slice()
          } else {
            chunk
          };

          match enc.encode(input_data, pkt.data.as_mut_slice()) {
            Ok(len) => {
              // let audio_samples = input_data.len() / channels;
              // let duration_samples = info.samples; // This is always frame_size for complete frames

              // // let duration_samples = audio_samples.min(self.frame_size);

              // // Convert sample duration to timebase units
              // let duration = if let Some(timebase) = frame.t.timebase {
              //   // Duration in timebase units
              //   duration_samples as u64
              // } else {
              //   // Fallback to samples
              //   duration_samples as u64
              // };

              // pkt.t.pts = Some(pts);
              // pkt.t.dts = Some(pts);
              // pkt.t.duration = Some(duration);
              // pkt.t.timebase = frame.t.timebase;

              // // Advance PTS by actual audio samples processed
              // // pts += duration_samples as i64;

              // pts += self.frame_size as i64;

              // pkt.data.truncate(len);
              // pending.push_back(pkt);

              // let duration =
              //   (Rational64::new(len as i64 / channels as i64, 48000)
              //     / frame.t.timebase.unwrap())
              //   .to_integer();
              let input_samples = input_data.len() / channels;
              let duration_samples = input_samples;
              pkt.t.pts = Some(pts);
              pkt.t.dts = Some(pts);
              pkt.t.duration = Some(duration_samples as u64);
              pkt.t.timebase = frame.t.timebase;

              // Always advance by frame_size (960 samples) for Opus
              pts += self.frame_size as i64;
              // pkt.t.pts = Some(pts);
              // pkt.t.dts = Some(pts);
              // pkt.t.duration = Some(duration as u64);
              // pts += duration;
              pkt.data.truncate(len);
              pending.push_back(pkt);
            }
            Err(_) => unimplemented!(),
          }
        }

        Ok(())
      } else {
        unimplemented!() // TODO mark it unreachable?
      }
    }

    fn receive_packet(&mut self) -> Result<Packet> {
      self.pending.pop_front().ok_or(Error::MoreDataNeeded)
    }

    fn set_option<'a>(&mut self, key: &str, val: Value<'a>) -> Result<()> {
      match (key, val) {
        // ("format", Value::Formaton(f)) => self.format = Some(f),
        // ("mapping", Value::ChannelMap(map) => self.cfg.map = map::to_vec()
        ("channels", Value::U64(v)) => self.cfg.channels = v as usize,
        ("streams", Value::U64(v)) => self.cfg.streams = v as usize,
        ("coupled_streams", Value::U64(v)) => {
          self.cfg.coupled_streams = v as usize
        }
        ("application", Value::Str(s)) => {
          if let Ok(a) = s.parse() {
            self.cfg.application = a;
          } else {
            return Err(Error::InvalidData);
          }
        }
        _ => return Err(Error::Unsupported("Unsupported option".to_owned())),
      }

      Ok(())
    }

    fn set_params(&mut self, params: &CodecParams) -> Result<()> {
      use av_data::params::*;
      if let Some(MediaKind::Audio(ref info)) = params.kind {
        if let Some(ref map) = info.map {
          if map.len() > 2 {
            unimplemented!()
          } else {
            self.cfg.channels = map.len();
            self.cfg.coupled_streams = self.cfg.channels - 1;
            self.cfg.streams = 1;
            self.cfg.mapping =
              if map.len() > 1 { vec![0, 1] } else { vec![0] };
          }
        }
      }
      Ok(())
    }

    // TODO: guard against calling it before configure()
    // is issued.
    fn get_params(&self) -> Result<CodecParams> {
      use av_data::params::*;
      use std::sync::Arc;
      Ok(CodecParams {
        kind: Some(MediaKind::Audio(AudioInfo {
          rate: 48000,
          map: Some(ChannelMap::default_map(1)),
          // Stereo
          // map: Some(ChannelMap::default_map(2)),
          format: Some(Arc::new(S16)),
        })),
        codec_id: Some("opus".to_owned()),
        extradata: self.get_extradata(),
        bit_rate: 16_000, // TODO: expose the information
        convergence_window: CONVERGENCE_WINDOW,
        delay: self.delay,
      })
    }

    fn flush(&mut self) -> Result<()> {
      // unimplemented!()
      self.flushing = true;
      Ok(())
    }
  }

  pub const OPUS_DESCR: &Des = &Des {
    descr: Descr {
      codec: "opus",
      name: "libopus",
      desc: "libopus encoder",
      mime: "audio/OPUS",
    },
  };
}

// pub use self::encoder_trait::OPUS_DESCR;

pub mod decoder_trait {
  use crate::OPUS_SET_GAIN_REQUEST;

  use super::Decoder as OpusDecoder;
  use av_bitstream::byteread::get_i16l;
  use av_codec::decoder::*;
  use av_codec::error::*;
  use av_data::audiosample::ChannelMap;
  use av_data::audiosample::formats::S16;
  use av_data::frame::*;
  use av_data::packet::Packet;
  use std::collections::VecDeque;
  use std::sync::Arc;

  pub struct Des {
    descr: Descr,
  }

  pub struct Dec {
    dec: Option<OpusDecoder>,
    extradata: Option<Vec<u8>>,
    pending: VecDeque<ArcFrame>,
    info: AudioInfo,
  }

  impl Dec {
    fn new() -> Self {
      Dec {
        dec: None,
        extradata: None,
        pending: VecDeque::with_capacity(1),
        info: AudioInfo {
          samples: 960 * 6,
          sample_rate: 48000,
          map: ChannelMap::new(),
          format: Arc::new(S16),
          block_len: None,
        },
      }
    }
  }

  impl Descriptor for Des {
    type OutputDecoder = Dec;

    fn create(&self) -> Self::OutputDecoder {
      Dec::new()
    }

    fn describe(&self) -> &Descr {
      &self.descr
    }
  }

  const OPUS_HEAD_SIZE: usize = 19;

  impl Decoder for Dec {
    fn set_extradata(&mut self, extra: &[u8]) {
      self.extradata = Some(Vec::from(extra));
    }
    fn send_packet(&mut self, pkt: &Packet) -> Result<()> {
      let mut f =
        Frame::new_default_frame(self.info.clone(), Some(pkt.t.clone()));

      let ret = {
        let buf: &mut [i16] = f.buf.as_mut_slice(0).unwrap();

        self
          .dec
          .as_mut()
          .unwrap()
          .decode(pkt.data.as_slice(), buf, false)
          .map_err(|_e| Error::InvalidData)
      };

      match ret {
        Ok(samples) => {
          if let MediaKind::Audio(ref mut info) = f.kind {
            info.samples = samples;
          }
          self.pending.push_back(Arc::new(f));
          Ok(())
        }
        Err(e) => Err(e),
      }
    }
    fn receive_frame(&mut self) -> Result<ArcFrame> {
      self.pending.pop_front().ok_or(Error::MoreDataNeeded)
    }
    fn configure(&mut self) -> Result<()> {
      let channels;
      let sample_rate = 48000;
      let mut gain_db = 0;
      let mut streams = 1;
      let mut coupled_streams = 0;
      let mut mapping: &[u8] = &[0u8, 1u8];
      let mut channel_map = false;

      if let Some(ref extradata) = self.extradata {
        // channels = *extradata.get(9).unwrap_or(&2) as usize;
        channels = *extradata.get(9).unwrap_or(&1) as usize;

        if extradata.len() >= OPUS_HEAD_SIZE {
          gain_db = get_i16l(&extradata[16..18]);
          channel_map = extradata[18] != 0;
        }
        if extradata.len() >= OPUS_HEAD_SIZE + 2 + channels {
          streams = extradata[OPUS_HEAD_SIZE] as usize;
          coupled_streams = extradata[OPUS_HEAD_SIZE + 1] as usize;
          if streams + coupled_streams != channels {
            unimplemented!()
          }
          mapping = &extradata[OPUS_HEAD_SIZE + 2..]
        } else {
          if channels > 2 || channel_map {
            return Err(Error::ConfigurationInvalid);
          }
          if channels > 1 {
            coupled_streams = 1;
          }
        }
      } else {
        return Err(Error::ConfigurationIncomplete);
      }

      if channels > 2 {
        unimplemented!() // TODO: Support properly channel mapping
      } else {
        self.info.map = ChannelMap::default_map(channels);
      }

      match OpusDecoder::create(
        sample_rate,
        channels,
        streams,
        coupled_streams,
        mapping,
      ) {
        Ok(mut d) => {
          let _ = d.set_option(OPUS_SET_GAIN_REQUEST, gain_db as i32);
          self.dec = Some(d);
          Ok(())
        }
        Err(_) => Err(Error::ConfigurationInvalid),
      }
    }

    fn flush(&mut self) -> Result<()> {
      self.dec.as_mut().unwrap().reset();
      Ok(())
    }
  }

  pub const OPUS_DESCR: &Des = &Des {
    descr: Descr {
      codec: "opus",
      name: "libopus",
      desc: "libopus decoder",
      mime: "audio/OPUS",
    },
  };
}

// pub use self::decoder_trait::OPUS_DESCR;
