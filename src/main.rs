use std::path::Path;

use hound::WavReader;
use rustfft::FFTplanner;
use rustfft::num_complex::Complex;

const PATH: &'static str = concat!(env!("CARGO_MANIFEST_DIR"), "/sin_440hz_44100hz_samp.wav");

fn find_spectral_peak<P: AsRef<Path>>(filename: P) -> Option<f32> {
    let mut reader = WavReader::open(filename).expect("failed to open wav file");
    let num_samples = reader.len() as usize;

    let mut planner = FFTplanner::new(false);
    let fft = planner.plan_fft(num_samples);

    let mut signal = reader.samples::<i16>()
        .map(|x| Complex::new(x.unwrap() as f32, 0f32))
        .collect::<Vec<_>>();

    let mut spectrum = signal.clone();
    fft.process(&mut signal[..], &mut spectrum[..]);

    // Only take the first half, the output is symmetrical, also Nyquist.
    let max_peak = spectrum.iter()
        .take(num_samples / 2)
        .enumerate()
        .max_by_key(|&(_, freq)| freq.norm() as u32);

    if let Some((i, _)) = max_peak {
        let bin = reader.spec().sample_rate as f32 / num_samples as f32;
        Some(i as f32 * bin)
    } else {
        None
    }
}

fn main() {
    println!("{:?}", find_spectral_peak(PATH));
}
