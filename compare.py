from config import cfg
def evaluate(scratch,path0,path1,mean,std,model,Thresh):
        
        cfg.merge_from_file(scratch)
        cfg.freeze()
        # Set the random seed manually for reproducibility.
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        #---------------------------

        wav_0 = audio.preprocess_wav(str(path0))
        wav_1 = audio.preprocess_wav(str(path1))

        fft_voice1 = audio.wav_to_spectrogram(wav_0)
        fft_voice2 = audio.wav_to_spectrogram(wav_1)

        input1=np.array(fft_voice1)
        input2=np.array(fft_voice2)

        #-------------------------
        test_dataset_verification = VoxcelebTestset(
            input1,input2,mean,std,cfg.DATASET.PARTIAL_N_FRAMES
        )

        #----data loader
        test_loader_verification = torch.utils.data.DataLoader(
            dataset=test_dataset_verification,
            batch_size=1,
            num_workers=cfg.DATASET.NUM_WORKERS,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )
        #---evaluate similarity 
        answer,score=validate_verification(cfg, model, test_loader_verification,Thresh)
        return answer,score
 
