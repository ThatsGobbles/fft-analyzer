use std::path::Path;

use hound::WavReader;
use rustfft::FFTplanner;
use rustfft::num_complex::Complex;

const PATH: &'static str = concat!(env!("CARGO_MANIFEST_DIR"), "/sin_440hz_44100hz_samp.wav");

const FFT_BUFFER_SIZE: usize = 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2;

fn _calculate_cutoff_freqs(
        num_bands: usize,
        global_lo_cutoff_freq: u32,  // In Hz
        global_hi_cutoff_freq: u32,  // In Hz
        sampling_freq: u32,  // In Hz
        effective_fft_output_len: usize,
        lo_cutoff_band_freqs: &mut [u32],  // In Hz
        hi_cutoff_band_freqs: &mut [u32],  // In Hz
        ) -> Result<(), &'static str> {
    if num_bands == 0 { Err("number of bands must be greater than 0")? }

    if num_bands != lo_cutoff_band_freqs.len() { Err("lcf length does not match")? }
    if num_bands != hi_cutoff_band_freqs.len() { Err("hcf length does not match")? }

    let freq_const =
        (global_lo_cutoff_freq as f32 / global_hi_cutoff_freq as f32).log10() /
        (1.0 / (num_bands as f32 + 1.0) - 1.0)
        // -(num_bands as f32 / (num_bands as f32 + 1.0))
    ;

    for n in 0..=num_bands {
        let power_of_ten = (((n as f32 + 1.0) / (num_bands as f32 + 1.0)) - 1.0) * freq_const;

        let fc = global_hi_cutoff_freq as f32 * 10.0f32.powf(power_of_ten);
        let fre = fc / (sampling_freq as f32 / 2.0);

        lo_cutoff_band_freqs[n] = (fre * effective_fft_output_len as f32) as u32 + 1;
        if n > 0 {
            if lo_cutoff_band_freqs[n] <= lo_cutoff_band_freqs[n - 1] {
                lo_cutoff_band_freqs[n] = lo_cutoff_band_freqs[n - 1] + 1;
            }

            hi_cutoff_band_freqs[n - 1] = lo_cutoff_band_freqs[n] - 1;
        }
    }

    Ok(())

    /*
    for (n = 0; n < bars + 1; n++) {
        double pot = freqconst * (-1);
        pot +=  ((float)n + 1) / ((float)bars + 1) * freqconst;
        fc[n] = p.highcf * pow(10, pot);
        fre[n] = fc[n] / (audio.rate / 2);
        //remember nyquist!, pr my calculations this should be rate/2
        //and  nyquist freq in M/2 but testing shows it is not...
        //or maybe the nq freq is in M/4

        //lfc stores the lower cut frequency foo each bar in the fft out buffer
        lcf[n] = fre[n] * (p.FFTbufferSize /2) + 1;
        if (n != 0) {
            hcf[n - 1] = lcf[n] - 1;

            //pushing the spectrum up if the expe function gets "clumped"
            if (lcf[n] <= lcf[n - 1])lcf[n] = lcf[n - 1] + 1;
            hcf[n - 1] = lcf[n] - 1;
        }
    }
    */
}

// int * separate_freq_bands(int FFTbufferSize, fftw_complex out[FFTbufferSize / 2 + 1],
// 			int bars, int lcf[200],  int hcf[200], double k[200], int channel,
// 			double sens, double ignore) {
// 	int o,i;
// 	double peak[201];
// 	static int fl[200];
// 	static int fr[200];
// 	double y[FFTbufferSize / 2 + 1];
// 	double temp;

// 	// process: separate frequency bands
// 	for (o = 0; o < bars; o++) {

// 		peak[o] = 0;

// 		// process: get peaks
// 		for (i = lcf[o]; i <= hcf[o]; i++) {
// 			y[i] = hypot(out[i][0], out[i][1]);
// 			peak[o] += y[i]; //adding upp band
// 		}

// 		peak[o] = peak[o] / (hcf[o]-lcf[o] + 1); //getting average
// 		temp = peak[o] * sens * k[o]; //multiplying with k and sens
// 		//printf("%d peak o: %f * sens: %f * k: %f = f: %f\n", o, peak[o], sens, k[o], temp);
// 		if (temp <= ignore) temp = 0;
// 		if (channel == 1) fl[o] = temp;
// 		else fr[o] = temp;

// 	}

// 	if (channel == 1) return fl;
//  	else return fr;
// }

fn separate_freq_bands(fft_output: &[Complex<f32>], num_bands: usize, output_buffer: &mut [f32]) -> Result<(), &'static str> {
    // Only take the first half, the output is symmetrical, also Nyquist.
    let effective_len = fft_output.len() / 2;

    // Make sure the output buffer is the same size as the effective length.
    if effective_len != output_buffer.len() { Err("buffer length does not match")? }

    Ok(())
}

fn find_spectral_peak<P: AsRef<Path>>(filename: P) -> Option<f32> {
    let mut reader = WavReader::open(filename).expect("failed to open wav file");
    let num_samples = reader.len() as usize;

    let mut planner = FFTplanner::new(false);
    let fft = planner.plan_fft(num_samples);

    let mut signal = reader.samples::<i32>()
        .map(|x| Complex::from(x.unwrap() as f32))
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
