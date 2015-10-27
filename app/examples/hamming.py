

Ts = 50e-6;                  % Sampling Time(s)
Fs = 1/Ts;                   % Sampling rate, Sampling Freq (Hz)
f0 = 50;                     % Frequency of interest (Hz)
duraT = 1;

%Calculate time axis
dt = 1/Fs;
tAxis = dt:dt:(duraT-dt);

y = cos(2*pi*f0*tAxis) +  2*sin(2*pi*10*tAxis);   y=y';

L = length (y); % Window Length of FFT    
nfft = 2^nextpow2(L); % Transform length

y_HannWnd = y.*hanning(L);            
Ydft_HannWnd = fft(y_HannWnd,nfft)/L;

   % at all frequencies except zero and the Nyquist
   mYdft = abs(Ydft_HannWnd);
   mYdft = mYdft (1:nfft/2+1);
   mYdft (2:end-1) = 2* mYdft(2:end-1);

f = Fs/2*linspace(0,1,nfft/2+1); 

  figure(1),
  subplot(2,1,1)
  plot(tAxis,y)
  title('Time Domain y(t)');
  xlabel('Time,s'); 
  ylabel('y');
  subplot(2,1,2)  
  plot(f,2*mYdft); % why need *2 ? Bcoz, Hanning Wnd Amplitude Correction Factor = 2
  axis ([0 500 0 5]); %Zoom in 
  title('Amplitude Spectrum with Hann Wnd');
  xlabel('Frequency (Hz)with hanning window'); 
  ylabel('|Y(f)|');