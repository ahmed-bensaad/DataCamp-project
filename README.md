# DataCamp-project : Traffic Signs Recognizer
Authors : Ahmed Ben Saad, Taha Halal, Yacine Ben Baccar

### To execute the script:

1. Install the `ramp-workflow` library (if not already done)

  ```
  $ pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git
  ```
  
2. Execute (if necessary) the scipt `annotations_gen.py` to generate `Train.py` and `Test.py`
  ```
  $ python annotations_gen.py
  ```
2. Launch the submission`main` 
  ```
  $ ramp_test_submission --submission main
  
  ```
2bis. For a quick training and testing of 1000 images for each step
  ```
  $ ramp_test_submission --submission main --quick-test
  ```
  
  ### Notebook
  
  Follow the provided in the jupyter notebook for some insights
  
  
  ### Dataset : The German Traffic Sign Recognition Benchmark  (GTSRB)
  
  The German Traffic Sign Benchmark is a multi-class, single-image classification challenge held at the International Joint Conference on Neural Networks (IJCNN) 2011. We cordially invite researchers from relevant fields to participate: The competition is designed to allow for participation without special domain knowledge.
  Check [this website](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) for further informations.



  #### Help
Go to the `ramp-workflow` [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki) for more help on the [RAMP](http:www.ramp.studio) ecosystem.
  
