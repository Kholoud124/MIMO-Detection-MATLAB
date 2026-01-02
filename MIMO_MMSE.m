clc; clear; close all;

%% MAIN SCRIPT 
num_tx = 3;
number_of_bits = 2e5;

%% Figure 1: BER vs Nr (SNR = 10 dB) 
SNR_fixed = 10;
Nr_range = 3:6;   

BER_ZF = zeros(size(Nr_range));
BER_MMSE = zeros(size(Nr_range));
BER_ZFSIC = zeros(size(Nr_range));
BER_MMSESIC = zeros(size(Nr_range));
BER_ML = zeros(size(Nr_range));

for k = 1:length(Nr_range)
    num_rx = Nr_range(k);

    BER_ZF(k)      = Zero_forcing(SNR_fixed, number_of_bits, num_tx, num_rx);
    BER_MMSE(k)    = mmse_equalizer(SNR_fixed, number_of_bits, num_tx, num_rx);
    BER_ZFSIC(k)   = zero_forcing_sic(SNR_fixed, number_of_bits, num_tx, num_rx);
    BER_MMSESIC(k) = mmse_sic(SNR_fixed, number_of_bits, num_tx, num_rx);
    BER_ML(k)      = ML_detector(SNR_fixed, number_of_bits, num_tx, num_rx);
end


figure(1)
semilogy(Nr_range, BER_ZF,'-o','LineWidth',1.5); hold on;
semilogy(Nr_range, BER_MMSE,'-s','LineWidth',1.5);
semilogy(Nr_range, BER_ZFSIC,'-*','LineWidth',1.5);
semilogy(Nr_range, BER_MMSESIC,'-x','LineWidth',1.5);
semilogy(Nr_range, BER_ML,'-d','LineWidth',1.5);

grid on;
xlabel('Number of Receive Antennas (Nr)');
ylabel('BER');
title('BER vs Nr at SNR = 10 dB');
legend('ZF','MMSE','ZF-SIC','MMSE-SIC','ML');

%% Figure 2: BER vs SNR (3x3 MIMO) 
num_rx = 3;
SNR_dB = 0:2:20;

BER2_ZF = zeros(size(SNR_dB));
BER2_MMSE = zeros(size(SNR_dB));
BER2_ZFSIC = zeros(size(SNR_dB));
BER2_MMSESIC = zeros(size(SNR_dB));
BER2_ML = zeros(size(SNR_dB));


for k = 1:length(SNR_dB)
    snr = SNR_dB(k);

    BER2_ZF(k)      = Zero_forcing(snr, number_of_bits, num_tx, num_rx);
    BER2_MMSE(k)    = mmse_equalizer(snr, number_of_bits, num_tx, num_rx);
    BER2_ZFSIC(k)   = zero_forcing_sic(snr, number_of_bits, num_tx, num_rx);
    BER2_MMSESIC(k) = mmse_sic(snr, number_of_bits, num_tx, num_rx);
    BER2_ML(k)      = ML_detector(snr, number_of_bits, num_tx, num_rx);
end


figure(2)
semilogy(SNR_dB, BER2_ZF,'-o','LineWidth',1.5); hold on;
semilogy(SNR_dB, BER2_MMSE,'-s','LineWidth',1.5);
semilogy(SNR_dB, BER2_ZFSIC,'-*','LineWidth',1.5);
semilogy(SNR_dB, BER2_MMSESIC,'-x','LineWidth',1.5);
semilogy(SNR_dB, BER2_ML,'-d','LineWidth',1.5);

grid on;
xlabel('SNR (dB)');
ylabel('BER');
title('BER vs SNR for 3×3 MIMO');
legend('ZF','MMSE','ZF-SIC','MMSE-SIC','ML');

%% FUNCTIONS 

function ber = mmse_equalizer(snr, number_of_bits, num_tx, num_rx)
bits_per_slot = 2*num_tx;
number_of_bits = number_of_bits - mod(number_of_bits,bits_per_slot);
data = randi([0 1],1,number_of_bits);

s = reshape(data,2,[]);
s = (2*s-1);
tx = (s(1,:)+1j*s(2,:))/sqrt(2);
tx = reshape(tx,num_tx,[]);

H = (randn(num_rx,num_tx,size(tx,2))+1j*randn(num_rx,num_tx,size(tx,2)))/sqrt(2);
snrL = 10^(snr/10);
nv = 1/(2*snrL);
n = sqrt(nv)*(randn(num_rx,size(tx,2))+1j*randn(num_rx,size(tx,2)));

rx = zeros(num_tx,size(tx,2));
for k=1:size(tx,2)
    y = H(:,:,k)*tx(:,k)+n(:,k);
    W = (H(:,:,k)'*H(:,:,k)+nv*eye(num_tx))\H(:,:,k)';
    rx(:,k)=W*y;
end

rx = rx(:).';
bits_hat = zeros(1,number_of_bits);
bits_hat(1:2:end)=real(rx)>0;
bits_hat(2:2:end)=imag(rx)>0;
ber = mean(bits_hat~=data);
end


%% MMSE_SIC
function ber = mmse_sic(snr, number_of_bits, num_tx, num_rx)
bits_per_slot = 2*num_tx;
number_of_bits = number_of_bits - mod(number_of_bits,bits_per_slot);
data = randi([0 1],1,number_of_bits);

s = reshape(data,2,[]);
s = (2*s-1);
tx = (s(1,:)+1j*s(2,:))/sqrt(2);
tx = reshape(tx,num_tx,[]);

H = (randn(num_rx,num_tx,size(tx,2))+1j*randn(num_rx,num_tx,size(tx,2)))/sqrt(2);
snrL = 10^(snr/10);
nv = 1/(2*snrL);
n = sqrt(nv)*(randn(num_rx,size(tx,2))+1j*randn(num_rx,size(tx,2)));

rx = zeros(num_tx,size(tx,2));
for t=1:size(tx,2)
    Hr=H(:,:,t); yr=Hr*tx(:,t)+n(:,t);
    det=zeros(num_tx,1); idx=1:num_tx;
    for k=1:num_tx
        W=(Hr'*Hr+nv*eye(size(Hr,2)))\Hr';
        [~,b]=min(sum(abs(W).^2,2));
        sh=W(b,:)*yr;
        sh=sign(real(sh))+1j*sign(imag(sh)); sh=sh/sqrt(2);
        det(idx(b))=sh;
        yr=yr-Hr(:,b)*sh;
        Hr(:,b)=[]; idx(b)=[];
    end
    rx(:,t)=det;
end

rx=rx(:).';
bits_hat=zeros(1,number_of_bits);
bits_hat(1:2:end)=real(rx)>0;
bits_hat(2:2:end)=imag(rx)>0;
ber=mean(bits_hat~=data);
end

%% ZF
function ber = Zero_forcing(snr, number_of_bits, num_tx, num_rx)
bits_per_slot = 2*num_tx;
number_of_bits = number_of_bits - mod(number_of_bits,bits_per_slot);
data=randi([0 1],1,number_of_bits);

s=reshape(data,2,[]);
s=(2*s-1);
tx=(s(1,:)+1j*s(2,:))/sqrt(2);
tx=reshape(tx,num_tx,[]);

H=(randn(num_rx,num_tx,size(tx,2))+1j*randn(num_rx,num_tx,size(tx,2)))/sqrt(2);
snrL=10^(snr/10);
nv=1/(2*snrL);
n=sqrt(nv)*(randn(num_rx,size(tx,2))+1j*randn(num_rx,size(tx,2)));

rx=zeros(num_tx,size(tx,2));
for k=1:size(tx,2)
    y=H(:,:,k)*tx(:,k)+n(:,k);
    rx(:,k)=pinv(H(:,:,k))*y;
end

rx=rx(:).';
bits_hat=zeros(1,number_of_bits);
bits_hat(1:2:end)=real(rx)>0;
bits_hat(2:2:end)=imag(rx)>0;
ber=mean(bits_hat~=data);
end

%% ZF_SIC
function ber = zero_forcing_sic(snr, number_of_bits, num_tx, num_rx)
bits_per_slot = 2*num_tx;
number_of_bits = number_of_bits - mod(number_of_bits,bits_per_slot);
data=randi([0 1],1,number_of_bits);

s=reshape(data,2,[]);
s=(2*s-1);
tx=(s(1,:)+1j*s(2,:))/sqrt(2);
tx=reshape(tx,num_tx,[]);

H=(randn(num_rx,num_tx,size(tx,2))+1j*randn(num_rx,num_tx,size(tx,2)))/sqrt(2);
snrL=10^(snr/10);
nv=1/(2*snrL);
n=sqrt(nv)*(randn(num_rx,size(tx,2))+1j*randn(num_rx,size(tx,2)));

rx=zeros(num_tx,size(tx,2));
for t=1:size(tx,2)
    Hr=H(:,:,t); yr=Hr*tx(:,t)+n(:,t);
    det=zeros(num_tx,1); idx=1:num_tx;
    for k=1:num_tx
        W=pinv(Hr);
        [~,b]=min(sum(abs(W).^2,2));
        sh=W(b,:)*yr;
        sh=sign(real(sh))+1j*sign(imag(sh)); sh=sh/sqrt(2);
        det(idx(b))=sh;
        yr=yr-Hr(:,b)*sh;
        Hr(:,b)=[]; idx(b)=[];
    end
    rx(:,t)=det;
end

rx=rx(:).';
bits_hat=zeros(1,number_of_bits);
bits_hat(1:2:end)=real(rx)>0;
bits_hat(2:2:end)=imag(rx)>0;
ber=mean(bits_hat~=data);
end
%% ML detection

function ber = ML_detector(snr, number_of_bits, num_tx, num_rx)

bits_per_slot = 2*num_tx;
number_of_bits = number_of_bits - mod(number_of_bits,bits_per_slot);
data = randi([0 1],1,number_of_bits);

s = reshape(data,2,[]);
s = (2*s-1);
tx = (s(1,:)+1j*s(2,:))/sqrt(2);
tx = reshape(tx,num_tx,[]);

H = (randn(num_rx,num_tx,size(tx,2))+1j*randn(num_rx,num_tx,size(tx,2)))/sqrt(2);
snrL = 10^(snr/10);
nv = 1/(2*snrL);
n = sqrt(nv)*(randn(num_rx,size(tx,2))+1j*randn(num_rx,size(tx,2)));

const = [1+1j 1-1j -1+1j -1-1j]/sqrt(2);
rx = zeros(num_tx,size(tx,2));

for t = 1:size(tx,2)
    y = H(:,:,t)*tx(:,t) + n(:,t);

    % All possible symbol combinations
    candidates = combvec(const,const,const);
    metric = zeros(1,size(candidates,2));

    for k = 1:size(candidates,2)
        metric(k) = norm(y - H(:,:,t)*candidates(:,k))^2;
    end

    [~,idx] = min(metric);
    rx(:,t) = candidates(:,idx);
end

rx = rx(:).';
bits_hat = zeros(1,number_of_bits);
bits_hat(1:2:end)=real(rx)>0;
bits_hat(2:2:end)=imag(rx)>0;

ber = mean(bits_hat~=data);
end
