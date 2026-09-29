/*
 * AudioWorklet that hands the page the raw microphone samples (mixed to mono)
 * in blocks of 4096, so live windows can be cut without gaps. It runs on the
 * browser's audio thread; the page decides what to do with the samples.
 */
class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.block = new Float32Array(4096);
    this.filled = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input.length) {
      const channels = input.length;
      for (let i = 0; i < input[0].length; i++) {
        let sum = 0;
        for (let c = 0; c < channels; c++) sum += input[c][i];
        this.block[this.filled++] = sum / channels;
        if (this.filled === this.block.length) {
          this.port.postMessage(this.block, [this.block.buffer]);
          this.block = new Float32Array(4096);
          this.filled = 0;
        }
      }
    }
    return true; // keep running
  }
}

registerProcessor("pcm-capture", PcmCapture);
