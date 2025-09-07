
testfile:
	ffmpeg -f lavfi -i "sine=frequency=440:duration=5" -acodec pcm_s16le -ar 48000 -ac 1 test.wav
