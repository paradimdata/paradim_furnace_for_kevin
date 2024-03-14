## Alex:  The Mask RCNN is in:
[Link to the matterport Mask RCNN github repo] (https://github.com/matterport/Mask_RCNN)

[Link to original reference by He et al 2017] (https://arxiv.org/abs/1703.06870)

For labeling the training data we use LabelMe which is available on Github here:  [Link to LabelMe](https://github.com/wkentaro/labelme)


On SciServer make a Compute container and pick the GPU domain (drop-down menu) with a TensorFlow image.

Nick will send you a link to his shared storage on SciServer that will have his most recent version running.  That will also have our training set so you can see what the jsons look like, etc.

We don't have TensorBoard running live on SciServer but you can download the trained HDF5 weights anywhere (it's in the *.h5 file in the logs directory) and then run TensorBoard locally which is an easy install.

---


# furnace_ml
Prototyping Mask R-CNN deployment for realtime melt-zone marking of furnace video. Push image frames to CasJobs, let CasJobs trigger masking on GPU Compute node.  Put the mask and image into Ali's table.  Matthew will dashboard displays frames from Ali's table.

Dependencies:
This repo contains Matterport's Mask_RCNN implementation as a submodule
git submodule add https://github.com/matterport/Mask_RCNN.git  -- no need to run this, I already have
If you clone this repo, you will need to init the mrcnn submodule:
From the project directory (furnace_ml), run:
git submodule init
git submodule update



clone repo
annotate and organize training data
run batch job
examine model
    need data from several runs
evaluate model


Goal: for each set of hyperparameters, perform a k-fold cross validation test. Store the k models for each cross-validation subtest, their scores (mean Average Precision, mean Average Recall, F1 score, loss?)
    
ParadimTrainDataOrganizer(Casjobs context, Casjobs tablename) - class for organizing/querying the paradim annotated-melted-zone data.

    InsertNewTrainingData(folder, class, growth? growthtime?)
        given a folder with .jpg + .json pairs and the associated class, inserts the data into CasJobs and places the file pairs into paradim_data volumne training data folder
        
    GenerateCrossValidationTrainingSets(k)
        k-fold cross validation. for a given k, will generate k training/validation datasets from the data recorded in the casjobs training table.  Each data element will serve in a validation set once. returns the folder containing the k train/val sets.
    
    PreprocessDataset(dataset folder)    
        Will apply some sort of preprocessing to a train/val dataset. ie, convert them all to black and white, etc
        
        
ParadimModel(Casjobs context, Casjobs tablename) - class for training/organizing/storing/evaluating a paradim crystal detection mr_cnn model 


(optional outer loop: over each training set)
For each set of hyperparameters
    create a f-fold cross-validation split of the dataset (f sets)
    For each cross-validation set
        train model
        eval model
            need confidence scores, intersection over union, class, ground truth class
    average eval scores over all cross-val sets/models
    Store hyper param score