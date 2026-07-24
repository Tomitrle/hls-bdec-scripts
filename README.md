# hls-bdec-scripts

# HLS Processing
The HLS_processing script is fairly straightforward to use. You should only have to set the tile_id, then the start_year and end_year, then the code should grab the relevant composite median 10-day HLS data (Red, NIR, Blue to calculate NDVI, EVI, and EVI2, QA to mask, and DOY to get the actual date) from the hls-composite-bdec bucket.

Note: the data requested may not exist in the bucket, i.e., if the tile_id is 17PPL, and the start_year is 2016, the bucket will only start downloading composites from 2020 and onwards, since that is the first year it has data for. If it appears that the bucket has no data for your tile, considering switching the prefix from f'step-2-evaluation-prototype-data/composite-median-10day/{tile_id}/' to f'step-1-evaluation-prototype-data/composite-median-10day/{tile_id}/'.

# MODIS Processing
The MODIS_processing script is a little more complicated, but the user input required is still relatively minimal. You will have to log in to use Earthdata. Then, set the MGRS_TILE, START_DATE, and END_DATE, similar to HLS Processing. If you want to generate a map of the ROI, there is code for that. Remember to change the roi_fname to properly indicate the source, and update the regional context section to describe the correct region. Otherwise, the following code will find the relevant HDF files (NDVI, EVI, Red and NIR to calculate EVI2, QA to mask, and DOY to get the actual date) that intersect the ROI (or tile if no valid ROI is provided) and convert them into GeoTIFFs.

Note: if the tifs appear to not have any valid VI pixels, the QA masking may be too strict, so you should consider increasing the threshold for usefulness or even removing it altogether. The same could be also done for the vi quality mask, but I would personally recommend leaving that one as is. For tropical areas with high cloud cover, it may also be necessary to remove some of the cloud masking. If you need to overwrite tif files, there is a check for if out_tif.exists() that you should comment out.

# HLS and MODIS Comparisons
This HLS_MODIS_comparisons is the most complicated script. First set the veg_index, tile_id, start_year, and end_year, like in the other scripts. Then run the respective compute_vi_stats functions, each of which will produce one dataframe and two (time, x, y) cubes (one for pixelwise vegetation index and one for pixelwise DOY)--to run them, you will have to uncomment or set the relevant ROI name and Shapefile.

The dataframe is used in the Timeseries section to create the timeseries and seasonal mean graphs. The cubes are used in the next phenometrics section.

For the phenometrics, you can change the sensor_config if you would like. The first one should already be set up for HLS, but if you want to try a different configuration, you can adjust the values. You should also check "n_knots = " in the code, since there are two values, one commented, and each corresponds with a different sensor (MODIS should be run with less knots). After the configuration and n_knots is ready, run the phenometrics code for the sensor that it is ready for. This will download annual tif files for the following metrics:

- mean_vi
- max_vi
- min_vi
- max_doy
- amplitude
- greenup_doy
- dormancy_doy
- growing_season_length
- min_doy
- greenup_vi
- dormancy_vi
- greenup_threshold
- auc_full
- auc_net
- greenup_rate
- greenup_rate_doy
- senescence_rate
- senescence_rate_doy
- mean_revisit_time
- quality_pixel_cnt

Then, change the sensor_config and n_knots for the other sensor and run that phenometrics call. Once you have the tif files, you can run plot_annual_phenometrics to visualize the rasters and/or run build_summary_df_from_tifs and _plot_phenology_summary to create line graphs comparing the annual phenometrics of HLS and MODIS.