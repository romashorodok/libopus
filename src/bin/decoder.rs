use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use av_format::demuxer::Context as DemuxerCtx;
use av_format::{
  buffer::AccReader,
  demuxer::Event,
  muxer::{self, Writer},
};

use av_bitstream::byteread::get_i16l;
use av_codec::decoder::*;
use av_codec::error::*;
use av_data::audiosample::ChannelMap;
use av_data::audiosample::formats::S16;
use av_data::frame::*;
use av_data::packet::Packet;
use std::collections::VecDeque;

use libopus::decoder_trait;
use matroska::{demuxer::MkvDemuxer, muxer::MkvMuxer};

fn main() {
  let file = std::fs::File::open("test.mkv").unwrap();

  let mut demuxer = DemuxerCtx::new(MkvDemuxer::new(), AccReader::new(file));

  println!("read headers: {:?}", demuxer.read_headers().unwrap());
  println!("global info: {:#?}", demuxer.info);

  let mut dec = decoder_trait::OPUS_DESCR.create();

  if let ref streams = demuxer.info.streams {
    if let Some(stream) = streams.first() {
      if let Some(ref extradata) = stream.params.extradata {
        println!("Setting extradata: {} bytes", extradata.len());
        dec.set_extradata(extradata);
      } else {
        println!("No extradata found in stream");
      }
    }
  }

  dec.configure().unwrap();

  // loop {
  //   match demuxer.read_event() {
  //     Ok(event) => {
  //       println!("event: {:?}", event);
  //       match event {
  //         Event::MoreDataNeeded(sz) => {
  //           println!("we needed more data: {} bytes", sz)
  //         }
  //         Event::NewStream(s) => println!("new stream :{:?}", s),
  //         Event::NewPacket(packet) => {
  //           // println!("writing packet {:?}", packet);
  //           // dec.send_packet(&packet).unwrap();
  //         }
  //         Event::Continue => {
  //           continue;
  //         }
  //         Event::Eof => {
  //           // dec.flush();
  //           break;
  //         }
  //         _ => break,
  //       }
  //     }
  //     Err(e) => {
  //       println!("error: {:?}", e);
  //       break;
  //     }
  //   }
  // }

  let mut packet_count = 0;
  let mut frame_count = 0;

  loop {
    match demuxer.read_event() {
      Ok(event) => {
        match event {
          Event::MoreDataNeeded(sz) => {
            println!("Need more data: {} bytes", sz);
          }
          Event::NewStream(s) => {
            println!("New stream: {:?}", s);
          }
          Event::NewPacket(packet) => {
            packet_count += 1;
            println!(
              "Packet {}: stream={}, size={} bytes, PTS={:?}",
              packet_count,
              packet.stream_index,
              packet.data.len(),
              packet.t.pts
            );

            // Send packet to decoder
            match dec.send_packet(&packet) {
              Ok(_) => {
                // Try to receive decoded frames
                while let Ok(frame) = dec.receive_frame() {
                  frame_count += 1;
                  if let MediaKind::Audio(ref info) = frame.kind {
                    println!(
                      "Frame {}: {} samples, {} channels, {} Hz",
                      frame_count,
                      info.samples,
                      info.map.len(),
                      info.sample_rate
                    );
                  }
                }
              }
              Err(e) => {
                println!("Failed to decode packet {}: {:?}", packet_count, e);
              }
            }
          }
          Event::Continue => continue,
          Event::Eof => {
            println!("End of file reached");
            // Flush the decoder
            dec.flush().unwrap();

            // Get any remaining frames
            while let Ok(frame) = dec.receive_frame() {
              frame_count += 1;
              if let MediaKind::Audio(ref info) = frame.kind {
                println!(
                  "Final frame {}: {} samples",
                  frame_count, info.samples
                );
              }
            }
            break;
          }
          _ => break,
        }
      }
      Err(e) => {
        println!("Demuxer error: {:?}", e);
        break;
      }
    }
  }

  println!(
    "Processed {} packets, decoded {} frames",
    packet_count, frame_count
  );
  // println!("test")
}
