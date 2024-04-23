# [Using Data Synthesis to Improve Forecasting Accuracy for  Greenhouse Optimisation](https://www.overleaf.com/project/65579eefc530df4aa4c1d490) #

This project examines the use of **Machine Learning** (ML) techniques in
Controlled Environment Agriculture (CEA) systems to optimise energy
consumption. It assesses the effectiveness of two data synthesis methods,
**Generative Adversarial Networks** (GANs) and **Variational Autoencoders** (VAEs),
in creating synthetic datasets that replicate environmental conditions within
greenhouses.
Both GANs and VAEs are assessed for their ability to generate realistic datasets
from limited data. The synthetic data is used to train regression models and
Convolutional Neural Networks, and their performance is analyzed against
baseline models using real data to improve forecasting accuracy. The results are
compared across testing strategies to understand the impact of model
architecture.
The results indicate that incorporating synthetic data into ML training can
enhance model performance. Models trained on a combination of real and
synthetic data generally showed reduced MSE compared to those trained only
with real data, suggesting that synthetic data can be beneficial for improving the
accuracy of forecasting models in CEA systems.

**File Structure:**
- `/data`
  - `/evaluation_results`
  - `/gan_synthetic_data`
  - `/models`
  - `/openmodelica`
  - `/training_data`
  - `/vae_synthetic_data`
- `/img`
  - Images used in the final report
- `/src`
  - `/data_generation`
  - `/data_processing`
  - `/evaluation`
  - `/openmodelica`

#### /data/OpenModelica ####
Contains 3 types of file, old simulation weather data:
 - SC_10Dec-22Nov.txt 
 - 10Dec-22Nov.csv
New simulation weather data provided by the NOAA:
 - SC_Jan2004-Jan2024.txt
 - SP_Jan2004-Jan2024.txt
 - Jan2004-Jan2024.txt
OpenModelica model results:
 - NOAA_10yr_reduced.csv
 - NOAA_res_dropped.csv
 - NOAA_res_reduced.csv

### Acknowledgments ###

This project was submitted by Matthew McCarthy for BSc Data Science and Analytics, supervised by Dr. Gregory Provan
