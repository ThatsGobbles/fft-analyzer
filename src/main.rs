use std::path::Path;

use hound::WavReader;
use rustfft::FFTplanner;
use rustfft::num_complex::Complex;

const PATH: &'static str = concat!(env!("CARGO_MANIFEST_DIR"), "/sin_440hz_44100hz_samp.wav");

const FFT_BUFFER_SIZE: usize = 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2;

fn _cava_calculate_cutoff_freqs(
        num_bands: usize,
        global_lo_cutoff_freq: u32,  // In Hz
        global_hi_cutoff_freq: u32,  // In Hz
        sampling_freq: u32,
        effective_fft_output_len: usize,
    ) -> (Vec<u32>, Vec<u32>)
{
    let freq_const =
        (global_lo_cutoff_freq as f32 / global_hi_cutoff_freq as f32).log10() /
        (1.0 / (num_bands as f32 + 1.0) - 1.0)
    ;

    println!("{}", freq_const);

    let mut lcfs = vec![];
    let mut hcfs = vec![];

    for n in 0..=num_bands {
        let power_of_ten = (((n as f32 + 1.0) / (num_bands as f32 + 1.0)) - 1.0) * freq_const;

        let fc = global_hi_cutoff_freq as f32 * 10.0f32.powf(power_of_ten);
        let fre = fc / (sampling_freq as f32 / 2.0);

        // println!("{}, {}, {}, {}", n, power_of_ten, fc, fre);

        let lcf = (fre * effective_fft_output_len as f32) as u32 + 1;
        lcfs.push(lcf);

        if n > 0 {
            if lcfs[n] <= lcfs[n - 1] {
                lcfs[n] = lcfs[n - 1] + 1;
            }

            hcfs.push(lcfs[n] - 1);
        }
    }

    (lcfs, hcfs)
}

fn _assign_fft_bins_to_bands(
    fft_buffer_size: usize,
    num_bands: u32,
    sampling_freq: u32,
    lo_cutoff_freq: u32,
    hi_cutoff_freq: u32,
) -> (Vec<f32>, Vec<f32>) {
    assert!(fft_buffer_size > 0);
    assert!(num_bands > 0);
    assert!(hi_cutoff_freq < sampling_freq / 2);
    assert!(lo_cutoff_freq < hi_cutoff_freq);

    // A.K.A. The frequency resolution.
    let freq_bin_size = sampling_freq as f32 / fft_buffer_size as f32;

    // Only take the first half of the frequency bins.
    // This due to the Nyquist–Shannon sampling theorem,
    // and the fact that the latter half represent negative frequencies.
    let freq_bins = (0..=(fft_buffer_size / 2))
        .into_iter()
        .map(|k| freq_bin_size * k as f32)
        .collect::<Vec<_>>()
    ;

    // Given the low cutoff frequency, and the number of desired bands, calculate a logarithmic spread.
    // Taken from https://stackoverflow.com/questions/7778271/logarithmically-spacing-number
    let delta_base = (hi_cutoff_freq as f32 / lo_cutoff_freq as f32).powf(1.0 / num_bands as f32);
    let mut band_intervals =
        (0..num_bands)
        .into_iter()
        .map(|i| lo_cutoff_freq as f32 * delta_base.powf(i as f32))
        .collect::<Vec<_>>()
    ;
    band_intervals.push(hi_cutoff_freq as f32);

    (freq_bins, band_intervals)
}

// If the DFT input consists of N samples that are sampled with frequency Fs,
// the output of the DFT corresponds to the frequencies F = [0, Fs/N, 2*Fs/N, ..., (N−1)*Fs/N].

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

    if global_lo_cutoff_freq >= global_hi_cutoff_freq { Err("lo cutoff frequency must be less than hi cutoff frequency")? }

    // Taken from https://stackoverflow.com/questions/7778271/logarithmically-spacing-number
    let delta_base = (global_hi_cutoff_freq as f32 / global_lo_cutoff_freq as f32).powf(1.0 / num_bands as f32);

    let endpoints =
        (0..=num_bands)
        .into_iter()
        .map(|i| global_lo_cutoff_freq as f32 * delta_base.powf(i as f32))
        .collect::<Vec<_>>()
    ;

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
    // println!("{:?}", find_spectral_peak(PATH));

    // let (lcfs, hcfs) = cava_calculate_cutoff_freqs(256, 5000, 10000, 44100, 256);
    // println!("{:?}", lcfs);
    // println!("{:?}", hcfs);

    let (f, e) = _assign_fft_bins_to_bands(128, 12, 44100, 20, 10000);
    println!("{} {:?}", f.len(), f);
    println!("{} {:?}", e.len(), e);
}
