## Mathematics for Machine Learning
https://github.com/mml-book/mml-book.github.io/blob/master/book/mml-book.pdf

## MIT 6.S191 — Introduction to Deep Learning
https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0

https://introtodeeplearning.com/



# Planning
- Change commands to just use args instead of config files
    - generate_data
    - train
    - inference
- Kaggle data
    - QUESTION: how do we know the actual anomalies?
        -> "Class" column, 0 or 1
- Output generated data to data directory
- Trainer should have a function to just read_next_training_set
    - lazy evaluation, doesn't do anything until data is used
    - reading from the training set should read one batch at a time

Real-time inference option
    - Event driven file detector that triggers inference job
    - Real time metric/monitoring/visualization
        - It displays the real data coming in, and the output of the inference service
        - QUESTION: What service to use? 
    - Separate data source service that places data into the storage location
    - Adapt the data source to be able to pull batches from the real kaggle dataset and spit it out to the storage location
    - Allow data generation to be real-time changeable
        - Change from sinusoidal to some other pattern, or maybe a different frequency
    - Add dynamic training loop
        - Find other relevant patterns to switch things up so dynamic training does something