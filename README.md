
# TODO:
- [ ] Add a miner.md to the docs directory
- [ ] Add a validator.md to the docs directory
- [*] Code review and cleanup
- [ ] to run and check multiple validators and miners

- [ ] update the dataset to fetch more data after the first 100 rows
- [ ] Check for duplicates in the jobs
- [ ] Move and Serve the synthetic data from the localDB or S3
- [ ] Add ```api.py``` and expose the endpoints for a webapp
- [ ] update the code to support Organic validation(via youtube URL, file upload or from an existing Subnet)
- [ ] Leaderboard and Benchmarking the performance
- [ ] adjust the reward function just for transcription and gender prediction
- [ ] work on reward function for the miner based of WER(Word Error Rate)
Ex: THEY ATTACKED AND REMOVED THE VOICES OF RESISTANCE FROM OUR RADIO AND STEVIE (**TV**) STATIONS, Here the miner should not be rewarded for the incorrect transcription of "TV"
 - [ ] work on reward function for the miner based on Punctuation,Spelling, **Time of the dialouge**.
 - [ ] **PHASE 2:** Dataset to contain more data and more diverse data (bg noise,bg music, voice overlays, etc)
 - [ ] Have our own MLmodel for transcription and gender prediction (combined)

