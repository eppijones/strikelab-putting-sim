let context: AudioContext | undefined;

/** Quiet, generated cues; no network download or autoplay requirement. */
export function golfSound(kind: "hit" | "cup") {
  try {
    context ??= new AudioContext();
    void context.resume().catch(() => {});
    const now = context.currentTime;
    const notes = kind === "hit" ? [180] : [523.25, 659.25, 783.99];
    notes.forEach((frequency, i) => {
      const voice = context!.createOscillator(),
        gain = context!.createGain();
      const start = now + i * 0.09,
        duration = kind === "hit" ? 0.1 : 0.45;
      voice.type = kind === "hit" ? "triangle" : "sine";
      voice.frequency.setValueAtTime(frequency, start);
      if (kind === "hit")
        voice.frequency.exponentialRampToValueAtTime(65, start + duration);
      gain.gain.setValueAtTime(0.0001, start);
      gain.gain.exponentialRampToValueAtTime(0.1, start + 0.008);
      gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);
      voice.connect(gain);
      gain.connect(context!.destination);
      voice.start(start);
      voice.stop(start + duration + 0.02);
      voice.onended = () => {
        voice.disconnect();
        gain.disconnect();
      };
    });
  } catch {
    /* Sound is optional on devices without Web Audio. */
  }
}
