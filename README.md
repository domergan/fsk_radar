# FSK radar using the Infineon BGT24ATR22

Python implementation of a BFSK radar using the BGT24ATR22-YPA Doppler radar evaluation board. Builds on the Time-Frequency-Range diagram TFRgram method described by Rogers (2007) and previously applied to traffic monitoring by Paulsson (2016).

## Example output (walking person in LOS)

![TFRgram](figures/TFRgram.pdf) \
![Velocity](figures/velocity_median_filter.pdf) \
![Range](figures/range_median_filter.pdf)

## References

- Rogers, D. (2007), Development and Testing of a Multiple Frequency Continuous Wave Radar for Target Detection and Classification
- Paulsson, T. (2016), Traffic Monitoring.
- Infineon Radar Development Kit (RDK), MATLAB/C++ reference implementation for the BGT24ATR22.
