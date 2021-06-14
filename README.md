# Typhoon Impact forecasting model

This tool was developed as trigger mechanism for the typhoon Early action protocol of the philipiness resdcross FbF project. The model will predict the potential damage of a typhoon before landfall, the prediction will be percentage of completely damaged houses per manuciplaity.
The tool is available under the [GPL license](https://github.com/rodekruis/Typhoon-Impact-based-forecasting-model/blob/master/LICENSE)

## Installation 

Install the Python environment and wgrib packages:
```
bash init_env.sh
bash install_wgrib2.sh
```

### R

Requires: R 4.1.0
Create an `R_LIBS_USER` environment variable for your R packages, then run
```
bash install_R_packages.sh
```


## Running 

```
source venv/bin/activate
python mainpipeline.py
```
