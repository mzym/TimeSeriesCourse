# PAMAP2_Dataset: Physical Activity Monitoring

## Формат данных

Synchronized and labeled raw data from all the sensors (3 IMUs and the HR-monitor) is merged into 1 data file per subject per session, available as text-files (.dat).

Each of the data-files contains 20 columns per row, the columns contain the following data:

-   1 timestamp (s)
-   2 activityID (see II.2. for the mapping to the activities)
-   3 heart rate (bpm)
-   4 temperature (°C)
-   5-7 3D-acceleration data (ms-2), scale: ±16g, resolution: 13-bit
-   8-10 3D-acceleration data (ms-2), scale: ±6g, resolution: 13-bit
-   11-13 3D-gyroscope data (rad/s)
-   14-16 3D-magnetometer data (μT)
-   17-20 orientation (invalid in this data collection)

Missing sensory data due to wireless data dropping: missing values are indicated with NaN. Since data is given every 0.01s (due to the fact, that the IMUs have a sampling frequency of 100Hz), and the sampling frequency of the HR-monitor was only approximately 9Hz, the missing HR-values are also indicated with NaN in the data-files.

### Идентификаторы активностей

-   1 lying
-   2 sitting
-   3 standing
-   4 walking
-   5 running
-   6 cycling
-   7 Nordic walking
-   9 watching TV
-   10 computer work
-   11 car driving
-   12 ascending stairs
-   13 descending stairs
-   16 vacuum cleaning
-   17 ironing
-   18 folding laundry
-   19 house cleaning
-   20 playing soccer
-   24 rope jumping
-   0 other (transient activities)

## Ссылки

1. A. Reiss and D. Stricker. Introducing a New Benchmarked Dataset for Activity Monitoring. The 16th IEEE International Symposium on Wearable Computers (ISWC), 2012.
2. A. Reiss and D. Stricker. Creating and Benchmarking a New Dataset for Physical Activity Monitoring. The 5th Workshop on Affect and Behaviour Related Assistance (ABRA), 2012.
