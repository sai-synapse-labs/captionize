
# TODO:
- [ ] Add a miner.md to the docs directory
- [ ] Add a validator.md to the docs directory
- [x] Code review and cleanup
- [ ] to run and check multiple validators and miners

- [x] update the dataset to fetch more data after the first 100 rows
- [x]Check for duplicates in the jobs
- [ ] Move and Serve the synthetic data from the localDB or S3
- [ ] Add ```api.py``` and expose the endpoints for a webapp
- [ ] update the code to support Organic validation(via youtube URL, file upload or from an existing Subnet)
- [ ] Leaderboard  and Benchmarking the performance
- [x] adjust the reward function just for transcription and gender prediction
- [x] work on reward function for the miner based of WER(Word Error Rate)
Ex: THEY ATTACKED AND REMOVED THE VOICES OF RESISTANCE FROM OUR RADIO AND STEVIE (**TV**) STATIONS, Here the miner should not be rewarded for the incorrect transcription of "TV"
 - [x] work on reward function for the miner based on Punctuation ,Spelling
 - [ ]work on  reward function based  **Time of the dialouge**(should done with manual validation).
 - [ ] **PHASE 2:** Dataset to contain more data and more diverse data (bg noise,bg music, voice overlays, etc)
 - [ ] Have our own MLmodel for transcription and gender prediction (combined)
 

# Going tasks

-[x]make sure first 100 tasks are done before feteching new jobs
-[ ]to check the reward function and to normalise the text output from ASR
-[ ]check set_weights function
-[ ]look into save_state function
