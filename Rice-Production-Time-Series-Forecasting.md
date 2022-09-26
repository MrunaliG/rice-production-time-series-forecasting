Rice Production \| Time Series Forecasting
================
Mrunali Ghelani

### 

#### Introduction

Agriculture, food, and related industries contribute to 5% of the US
Gross Domestic Product (GDP). Rice is one of the 5 major crops produced
by the US. 45-50% of the crop is exported each year, and the USA is the
5th largest exporter of rice.

Rice production forecasts help the USDA understand if there will be a
surplus or scarcity of rice, and the federal government can accordingly
draft agriculture policies and award subsidies to farmers to avoid such
a situation. Subsidies typically focus on stabilizing the food chain or
ensuring adequate income for farmers. This federal support of specific
agricultural crops determines their availability and consumption, which
in turn impacts the average American’s diet and nutrition.

Rice production is analyzed and forecasted in terms of yield, quantity
produced, and producer price index. For these 3 time series, we look at:

-   What does the future look like for rice production?

-   How is the latest trend different from the past?

-   What are the implications for our forecast on the agricultural
    policy and subsidies?

#### Yield

##### Data

Data description:

-   Time period: 1960 - 2020
-   Time interval: Year
-   Total observations: 60
-   No missing data

``` r
rice_yield <- readr::read_csv("FAOSTAT_data_5-3-2022_RICE_YIELD.csv") %>%
  select(Year, Value) %>%
  as_tsibble(index=Year)

rice_yield$Value <- as.numeric(rice_yield$Value)

glimpse(rice_yield)
```

    Rows: 60
    Columns: 2
    $ Year  <dbl> 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971…
    $ Value <dbl> 38227, 41785, 44449, 45906, 47693, 48442, 50851, 49600, 48399, 5…

``` r
count_gaps(rice_yield)
```

    # A tibble: 0 × 3
    # … with 3 variables: .from <dbl>, .to <dbl>, .n <int>

``` r
rice_yield %>% autoplot() + labs(title = "Rice Yield (1960-2020)", y="Yield (in hg/ha)")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

##### Train-Test Data Split

An 80-20 split is used for the training and test data

``` r
split_id = round(nrow(rice_yield) * 0.8)
split_year = rice_yield[split_id, 1]

rice_yield_train <- rice_yield %>%
  filter(Year <= split_year$Year)

print(paste("Training data observations = ", nrow(rice_yield_train)))
```

    [1] "Training data observations =  48"

``` r
rice_yield_test <- rice_yield %>% 
  filter(Year > split_year$Year)

print(paste("Test data observations = ", nrow(rice_yield_test)))
```

    [1] "Test data observations =  12"

##### Trend

``` r
rice_yield_train %>% 
  model(STL(Value)) %>%
  components() %>%
  autoplot() + labs(title = "STL Decomposition: Rice Yield", y="Yield (in hg/ha)")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

-   Upward trend
-   No seasonality as data is yearly
-   Remainder resembles white noise

##### ACF and PACF plots

``` r
rice_yield_train %>% 
  gg_tsdisplay(Value, plot_type="partial", lag_max=48) + labs(title = "ACF and PACF Plots: Rice Yield", y="Yield (in hg/ha)")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
rice_yield_train %>% 
  gg_tsdisplay(difference(Value), plot_type="partial", lag_max=48) + labs(title = "ACF and PACF plots: Rice Yield with Differentiation", y="Yield (in hg/ha)")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-5-2.png)<!-- -->

-   PACF plot has a significant spike at lag 1 only
-   ACF plot is sinusoidal, and multiple lags are significant
-   The differenced series resembles white noise, with the lags largely
    uncorrelated

Hence, ETS with trend component, MA(1) with lag-1 differencing and auto
ARIMA models are evaluated below

##### Model Creation and Training Accuracy

``` r
fit <- rice_yield_train %>%
  model(ets = ETS(Value),
        arima = ARIMA(Value ~ pdq(0, 1, 1)),
        auto = ARIMA(Value))

fit %>% 
  select(arima) %>%
  report()
```

    Series: Value 
    Model: ARIMA(0,1,1) w/ drift 

    Coefficients:
              ma1  constant
          -0.3935  820.6476
    s.e.   0.1438  229.4626

    sigma^2 estimated as 6826051:  log likelihood=-435.55
    AIC=877.11   AICc=877.67   BIC=882.66

``` r
fit %>%
  select(auto) %>%
  report()
```

    Series: Value 
    Model: ARIMA(0,1,1) w/ drift 

    Coefficients:
              ma1  constant
          -0.3935  820.6476
    s.e.   0.1438  229.4626

    sigma^2 estimated as 6826051:  log likelihood=-435.55
    AIC=877.11   AICc=877.67   BIC=882.66

``` r
fit %>% accuracy()
```

    # A tibble: 3 × 10
      .model .type        ME  RMSE   MAE     MPE  MAPE  MASE RMSSE    ACF1
      <chr>  <chr>     <dbl> <dbl> <dbl>   <dbl> <dbl> <dbl> <dbl>   <dbl>
    1 ets    Training -386.  2631. 2144. -0.888   3.71 0.871 0.926 -0.0363
    2 arima  Training   39.3 2530. 2098. -0.0168  3.59 0.852 0.890  0.0241
    3 auto   Training   39.3 2530. 2098. -0.0168  3.59 0.852 0.890  0.0241

Based on the training accuracy, ARIMA(0, 1, 1) is the best model

##### Forecasts and Forecast Accuracy

``` r
fc <- forecast(fit, h=nrow(rice_yield_test)) 

fc %>% accuracy(rice_yield_test)
```

    # A tibble: 3 × 10
      .model .type     ME  RMSE   MAE   MPE  MAPE  MASE RMSSE  ACF1
      <chr>  <chr>  <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
    1 arima  Test   -793. 2590. 2159. -1.02  2.63   NaN   NaN 0.370
    2 auto   Test   -793. 2590. 2159. -1.02  2.63   NaN   NaN 0.370
    3 ets    Test  -1381. 2929. 2519. -1.72  3.06   NaN   NaN 0.397

``` r
fc %>% 
  filter(.model %in% c("arima", "ets")) %>%
  autoplot(rice_yield) + labs(title = "Forecasts: Rice Yield", y="Yield (in hg/ha)")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

Based on the forecast accuracy and plot, ARIMA(0, 1, 1) is the best
model

##### Residual Analysis

``` r
fit %>% 
  augment() %>% 
  features(.innov, box_pierce, lag = 10, dof = 2)
```

    # A tibble: 3 × 3
      .model bp_stat bp_pvalue
      <chr>    <dbl>     <dbl>
    1 arima     6.10     0.636
    2 auto      6.10     0.636
    3 ets       5.03     0.754

``` r
fit %>% 
  augment() %>% 
  features(.innov, ljung_box, lag = 10, dof = 2)
```

    # A tibble: 3 × 3
      .model lb_stat lb_pvalue
      <chr>    <dbl>     <dbl>
    1 arima     7.57     0.476
    2 auto      7.57     0.476
    3 ets       6.12     0.634

``` r
fit %>% 
  select(arima) %>%
  gg_tsresiduals() + labs(title = "Residual Analysis (ARIMA): Rice Yield")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

For the ARIMA(0, 1, 1) model, the residuals are uncorrelated, mean-0
centered and do not have a normal distribution. Hence, we can use the
point forecasts but the prediction interval forecasts may not be
accurate

This confirms that the model is fairly good for forecasting rice yield

##### Five-Year Forecast

``` r
fit <- rice_yield %>%
  model(arima = ARIMA(Value ~ pdq(0, 1, 1)))

fc <- forecast(fit, h=5) 

fc %>% autoplot(rice_yield) + labs(title = "Five-Year Forecast: Rice Yield", y="Yield (in hg/ha)")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

``` r
fc
```

    # A fable: 5 x 4 [1Y]
    # Key:     .model [1]
      .model  Year             Value  .mean
      <chr>  <dbl>            <dist>  <dbl>
    1 arima   2021 N(86242, 6897818) 86242.
    2 arima   2022 N(87018, 9441395) 87018.
    3 arima   2023 N(87794, 1.2e+07) 87794.
    4 arima   2024 N(88570, 1.5e+07) 88570.
    5 arima   2025 N(89346, 1.7e+07) 89346.

Five-Year Point Forecasts:

-   2021: 86,242 hg/ha
-   2022: 87,018 hg/ha
-   2023: 87,794 hg/ha
-   2024: 88,570 hg/ha
-   2025: 89,346 hg/ha

The forecasts follow a continuous upward trend, with year-on-year growth
of 1%

#### Production Quantity

##### Data

Data description:

-   Time period: 1960 - 2020
-   Time interval: Year
-   Total observations: 60
-   No missing data

``` r
rice_produced_qty <- readr::read_csv("FAOSTAT_data_5-3-2022_RICE_PROD_QTY.csv") %>%
  select(Year, Value) %>%
  as_tsibble(index=Year)

rice_produced_qty$Value <- as.numeric(rice_produced_qty$Value)

glimpse(rice_produced_qty)
```

    Rows: 60
    Columns: 2
    $ Year  <dbl> 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971…
    $ Value <dbl> 2458000, 2996000, 3187000, 3319000, 3460000, 3856422, 4054142, 4…

``` r
count_gaps(rice_produced_qty)
```

    # A tibble: 0 × 3
    # … with 3 variables: .from <dbl>, .to <dbl>, .n <int>

``` r
rice_produced_qty %>% 
  autoplot() + labs(title = "Rice Production Quantity (1960-2020)", y="Quantity (in tonnes)")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

##### Train-Test data split

An 80-20 split is used for the training and test data

``` r
split_id = round(nrow(rice_produced_qty) * 0.8)
split_year = rice_produced_qty[split_id, 1]

rice_produced_qty_train <- rice_produced_qty %>%
  filter(Year <= split_year$Year)

print(paste("Training data observations = ", nrow(rice_produced_qty_train)))
```

    [1] "Training data observations =  48"

``` r
rice_produced_qty_test <- rice_produced_qty %>% 
  filter(Year > split_year$Year)

print(paste("Test data observations = ", nrow(rice_produced_qty_test)))
```

    [1] "Test data observations =  12"

##### Trend

``` r
rice_produced_qty_train %>% 
  model(STL(Value)) %>%
  components() %>%
  autoplot() + labs(title = "STL Decomposition: Rice Production Quantity", y="Quantity (in tonnes)")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

-   Upward trend
-   No seasonality as data is yearly
-   Remainder resembles white noise

##### ACF and PACF plots

``` r
rice_produced_qty_train %>%
  gg_tsdisplay(Value, plot_type="partial", lag_max=48) + labs(title = "ACF and PACF Plots: Rice Production Quantity", y="Quantity (in tonnes))")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

``` r
rice_produced_qty_train %>%
  gg_tsdisplay(difference(Value), plot_type="partial", lag_max=48) + labs(title = "ACF and PACF Plots: Rice Production Quantity with Differentiation", y="Quantity (in tonnes))")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-13-2.png)<!-- -->

-   PACF plot has a significant spike at lag 1 only
-   ACF plot is sinusoidal, and multiple lags are significant
-   The differenced series resembles white noise, with the lags largely
    uncorrelated (most significant spike for ACF plot at lag-10 and for
    PACF plot at lag-2)

Hence, ETS with trend component, MA(1) with lag-1 differencing and auto
ARIMA models are evaluated below

##### Model Creation and Training Accuracy

``` r
fit <- rice_produced_qty_train %>%
  model(ets = ETS(Value),
        arima = ARIMA(Value ~ pdq(0, 1, 1)),
        auto = ARIMA(Value))

fit %>% 
  select(arima) %>%
  report()
```

    Series: Value 
    Model: ARIMA(0,1,1) 

    Coefficients:
              ma1
          -0.4231
    s.e.   0.1333

    sigma^2 estimated as 6.814e+11:  log likelihood=-706.6
    AIC=1417.19   AICc=1417.47   BIC=1420.89

``` r
fit %>%
  select(auto) %>%
  report()
```

    Series: Value 
    Model: ARIMA(2,1,0) w/ drift 

    Coefficients:
              ar1     ar2  constant
          -0.4328  -0.347  247905.4
    s.e.   0.1355   0.133  113126.8

    sigma^2 estimated as 6.251e+11:  log likelihood=-703.61
    AIC=1415.22   AICc=1416.17   BIC=1422.62

``` r
fit %>% accuracy()
```

    # A tibble: 3 × 10
      .model .type          ME    RMSE     MAE    MPE  MAPE  MASE RMSSE    ACF1
      <chr>  <chr>       <dbl>   <dbl>   <dbl>  <dbl> <dbl> <dbl> <dbl>   <dbl>
    1 ets    Training -100992. 758747. 555280. -2.61   9.17 0.842 0.864  0.0476
    2 arima  Training  241138. 808069. 596401.  3.42   9.72 0.905 0.920 -0.0691
    3 auto   Training    4818. 756984. 562909. -0.539  9.10 0.854 0.862 -0.0360

Based on the training accuracy, ARIMA(2, 1, 0) is the best model

##### Forecasts and Forecast Accuracy

``` r
fc <- forecast(fit, h=nrow(rice_produced_qty_test))

fc %>% accuracy(rice_produced_qty_test)
```

    # A tibble: 3 × 10
      .model .type        ME     RMSE      MAE    MPE  MAPE  MASE RMSSE    ACF1
      <chr>  <chr>     <dbl>    <dbl>    <dbl>  <dbl> <dbl> <dbl> <dbl>   <dbl>
    1 arima  Test    215747.  954821.  870224.   1.33  9.13   NaN   NaN -0.510 
    2 auto   Test   -677101. 1291426. 1056460.  -8.31 11.9    NaN   NaN -0.111 
    3 ets    Test  -1326886. 1781347. 1553681. -15.3  17.4    NaN   NaN  0.0282

``` r
fc %>% autoplot(rice_produced_qty) + labs(title = "Forecasts: Rice Production Quantity", y="Quantity (in tonnes)")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

Based on the forecast accuracy and plot, ARIMA(0, 1, 1) is the best
model

##### Residual Analysis

``` r
fit %>% 
  augment() %>% 
  features(.innov, box_pierce, lag = 10, dof = 2)
```

    # A tibble: 3 × 3
      .model bp_stat bp_pvalue
      <chr>    <dbl>     <dbl>
    1 arima    14.0     0.0808
    2 auto      9.32    0.316 
    3 ets      15.2     0.0549

``` r
fit %>% 
  augment() %>% 
  features(.innov, ljung_box, lag = 10, dof = 2)
```

    # A tibble: 3 × 3
      .model lb_stat lb_pvalue
      <chr>    <dbl>     <dbl>
    1 arima     17.0    0.0300
    2 auto      11.4    0.181 
    3 ets       18.3    0.0193

``` r
fit %>% 
  select(arima) %>%
  gg_tsresiduals() + labs(title = "Residual Analysis (ARIMA): Rice Production Quantity")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

For the ARIMA(0, 1, 1) model, the residuals resemble white noise based
on Box-Pierce test, are largely uncorrelated, close to mean-0 centered
and do not have a normal distribution. Hence, we may use the point
forecasts but the prediction interval forecasts may not be accurate

This confirms that the model is fairly good for forecasting rice
production quantity

##### Five-Year Forecast

``` r
fit <- rice_produced_qty %>%
  model(arima = ARIMA(Value ~ pdq(0, 1, 1)))

fc <- forecast(fit, h=5) 

fc %>% autoplot(rice_produced_qty) + labs(title = "Five-Year Forecast: Rice Production Quantity", y="Quantity (in tonnes)")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

``` r
fc
```

    # A fable: 5 x 4 [1Y]
    # Key:     .model [1]
      .model  Year               Value     .mean
      <chr>  <dbl>              <dist>     <dbl>
    1 arima   2021 N(9869139, 7.4e+11)  9869139.
    2 arima   2022     N(1e+07, 8e+11)  9985716.
    3 arima   2023   N(1e+07, 8.5e+11) 10102293.
    4 arima   2024   N(1e+07, 9.1e+11) 10218870.
    5 arima   2025   N(1e+07, 9.7e+11) 10335447.

Five-Year Point Forecasts:

-   2021: 9,869,139 tonnes
-   2022: 9,985,716 tonnes
-   2023: 10,102,293 tonnes
-   2024: 10,218,870 tonnes
-   2025: 10,335,447 tonnes

The forecasts follow a continuous upward trend, with year-on-year growth
of 1%

#### Producer Price Index (YoY)

##### Data

Data description:

-   Series ID and link:
    [WPU01230103](https://fred.stlouisfed.org/series/WPU01230103)
-   Time period: 2010 Jan - 2022 Mar
-   Time interval: Month
-   Total observations: 135
-   No missing data
-   Month-on-Month % change series resembles a white noise series with
    mean close to 0 and hence cannot be used for forecasting
-   Year-on-Year % change series is used for further analysis

``` r
rice_ppi <- readr::read_csv("WPU01230103_RICE_PPI.csv") %>%
  mutate('MonthYear' = yearmonth(DATE)) %>%
  rename(Value = WPU01230103) %>%
  select(MonthYear, Value) %>%
  as_tsibble(index=MonthYear) %>%
  filter(MonthYear >= yearmonth("2010 Jan"))
  
rice_ppi$Value <- as.numeric(rice_ppi$Value)

rice_ppi <- rice_ppi %>%
  mutate(MoM = (Value - lag(Value)) / lag(Value) * 100,
    YoY = (Value - lag(Value, 12)) / lag(Value, 12) * 100) 

rice_ppi <- rice_ppi %>% drop_na()

nrow(rice_ppi)
```

    [1] 135

``` r
glimpse(rice_ppi)
```

    Rows: 135
    Columns: 4
    $ MonthYear <mth> 2011 Jan, 2011 Feb, 2011 Mar, 2011 Apr, 2011 May, 2011 Jun, …
    $ Value     <dbl> 173.6, 177.7, 175.0, 175.0, 166.9, 158.8, 170.9, 183.0, 192.…
    $ MoM       <dbl> 0.7544980, 2.3617512, -1.5194147, 0.0000000, -4.6285714, -4.…
    $ YoY       <dbl> -10.4231166, -8.3075335, -7.1125265, -5.7619817, -8.1452944,…

``` r
count_gaps(rice_ppi)
```

    # A tibble: 0 × 3
    # … with 3 variables: .from <mth>, .to <mth>, .n <int>

``` r
rice_ppi %>% autoplot(Value) + labs(title = "Rice Production Price Index (2010-2022)")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

``` r
rice_ppi %>% autoplot(MoM) + labs(title = "Rice PPI Month-on-Month % Change", y="% change")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-18-2.png)<!-- -->

``` r
rice_ppi %>% autoplot(YoY) + labs(title = "Rice PPI Year-on-Year % Change", y="% change")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-18-3.png)<!-- -->

##### Train-Test data split

An 80-20 split is used for the training and test data

``` r
split_id = round(nrow(rice_ppi) * 0.8)
split_yearmonth = rice_ppi[split_id, 1]

rice_ppi_train <- rice_ppi %>%
  filter(MonthYear <= split_yearmonth$MonthYear)

print(paste("Training data observations = ", nrow(rice_ppi_train)))
```

    [1] "Training data observations =  108"

``` r
rice_ppi_test <- rice_ppi %>% 
  filter(MonthYear > split_yearmonth$MonthYear)

print(paste("Test data observations = ", nrow(rice_ppi_test)))
```

    [1] "Test data observations =  27"

##### Seasonality & Trend

``` r
rice_ppi_train %>% 
  model(STL(YoY)) %>%
  components() %>%
  autoplot() + labs(title = "STL Decomposition: Rice Production Price Index (YoY)", y="% change")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

``` r
rice_ppi_train %>%
  gg_season(YoY) + labs(y="", title="Seasonal Plot: Rice Production Price Index (YoY)")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-20-2.png)<!-- -->

-   No seasonality
-   No overall trend (Trend patterns fluctuate with time around mean-0)
-   Remainder is mean-0 centered with non-constant variance

##### ACF and PACF plots

``` r
rice_ppi_train %>% 
  gg_tsdisplay(YoY, plot_type="partial", lag_max=24) + labs(title = "ACF and PACF Plots: Rice Production Price Index (YoY)")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-21-1.png)<!-- -->

-   PACF plot has a significant spike at lag 1, with smaller spikes at
    lags 2, 10, 13
-   ACF plot is sinusoidal, and multiple lags are significant

Hence, MA(1)/MA(2) and auto ARIMA models are evaluated below

##### Model Creation and Training Accuracy

``` r
fit <- rice_ppi_train %>%
  model(arima = ARIMA(YoY ~ pdq(0, 1:13, 1:2) + PDQ(0, 0, 0)),
        auto = ARIMA(YoY))

fit %>% 
  select(arima) %>% 
  report()
```

    Series: YoY 
    Model: ARIMA(0,1,2) 

    Coefficients:
             ma1     ma2
          0.4040  0.4575
    s.e.  0.0875  0.1086

    sigma^2 estimated as 20.08:  log likelihood=-311.57
    AIC=629.13   AICc=629.36   BIC=637.15

``` r
fit %>% 
  select(auto) %>% 
  report()
```

    Series: YoY 
    Model: ARIMA(1,0,2)(2,0,0)[12] 

    Coefficients:
             ar1     ma1     ma2     sar1     sar2
          0.9176  0.3446  0.3456  -0.6183  -0.1795
    s.e.  0.0418  0.1061  0.1102   0.1098   0.1106

    sigma^2 estimated as 13.94:  log likelihood=-296.61
    AIC=605.22   AICc=606.05   BIC=621.32

``` r
fit %>% accuracy()
```

    # A tibble: 2 × 10
      .model .type        ME  RMSE   MAE   MPE  MAPE  MASE RMSSE    ACF1
      <chr>  <chr>     <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>   <dbl>
    1 arima  Training 0.0878  4.42  3.33   Inf   Inf 0.198 0.210 -0.0507
    2 auto   Training 0.0765  3.65  2.77   Inf   Inf 0.165 0.173 -0.0123

Based on the training accuracy, ARIMA(1, 0, 2)(2, 0, 0) is the best
model

##### Forecasts and Forecast Accuracy

``` r
fc <- forecast(fit, h=nrow(rice_ppi_test)) 

fc %>% accuracy(rice_ppi_test)
```

    # A tibble: 2 × 10
      .model .type    ME  RMSE   MAE    MPE  MAPE  MASE RMSSE  ACF1
      <chr>  <chr> <dbl> <dbl> <dbl>  <dbl> <dbl> <dbl> <dbl> <dbl>
    1 arima  Test   1.57  6.01  5.04 -464.  592.    NaN   NaN 0.773
    2 auto   Test   7.39  8.79  7.44   92.5  92.5   NaN   NaN 0.644

``` r
fc %>% autoplot(rice_ppi) + labs(title = "Forecasts: Rice Production Price Index")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-23-1.png)<!-- -->

Based on the forecast accuracy and plot, ARIMA(0, 1, 2) is the better
model. However, ARIMA(1, 0, 2)(2, 0, 0) model provides good forecasts as
well, with smaller prediction intervals.

##### Residual Analysis

``` r
fit %>% 
  augment() %>%
  features(.innov, box_pierce, lag = 24, dof = 2)
```

    # A tibble: 2 × 3
      .model bp_stat bp_pvalue
      <chr>    <dbl>     <dbl>
    1 arima     38.2    0.0173
    2 auto      17.4    0.742 

``` r
fit %>% 
  augment() %>%
  features(.innov, ljung_box, lag = 24, dof = 2)
```

    # A tibble: 2 × 3
      .model lb_stat lb_pvalue
      <chr>    <dbl>     <dbl>
    1 arima     44.1   0.00345
    2 auto      20.6   0.544  

``` r
fit %>% 
  select(auto) %>%
  gg_tsresiduals() + labs(title = "Residual Analysis (ARIMA): Rice Production Price Index")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-24-1.png)<!-- -->

For the ARIMA(1, 0, 2)(2, 0, 0) model, the residuals resemble white
noise, are uncorrelated, mean-0 centered and have a close to normal
distribution. Hence, the model suggests that we may use the point
forecasts but the prediction interval forecasts may not be reliable

Hence, we use ARIMA(1, 0, 2)(2, 0, 0) model for forecasting.

For better forecasts, it may be advisable to experiment with non-linear
models, or conduct sub-sampling analysis (pre-pandemic and post-pandemic
time series analysis)

##### 6-Month Forecast

``` r
fit <- rice_ppi %>%
  model(arima = ARIMA(YoY ~ pdq(1, 0, 2) + PDQ(2, 0, 0)))

fc <- forecast(fit, h=6) 

fc %>% autoplot(rice_ppi) + labs(title = "6-Month Forecast: Rice Producer Price Index (YoY)", y = "% change")
```

![](Rice-Production-Time-Series-Forecasting_files/figure-gfm/unnamed-chunk-25-1.png)<!-- -->

``` r
fc
```

    # A fable: 6 x 4 [1M]
    # Key:     .model [1]
      .model MonthYear        YoY .mean
      <chr>      <mth>     <dist> <dbl>
    1 arima   2022 Apr  N(21, 13)  21.0
    2 arima   2022 May  N(20, 32)  20.1
    3 arima   2022 Jun  N(21, 60)  20.6
    4 arima   2022 Jul  N(19, 84)  19.2
    5 arima   2022 Aug N(17, 105)  16.8
    6 arima   2022 Sep N(13, 124)  13.0

Six-Month Point Forecasts:

-   2022 Apr: **21%**
-   2022 May: **20.1%**
-   2022 Jun: **20.6%**
-   2022 Jul: **19.2%**
-   2022 Aug: **16.8%**
-   2022 Sep: **13%**

The model forecasts that the producer price index will continue to
increase as compared to the previous year, however, this YoY % change
will decrease over time. It predicts that the highest YoY % change will
be seen in April 2022 of **21%**, with predicted PPI as **217.8**

#### Economic Outlook and Policy

Rice yield has increased over the years and we forecast that it will
continue to increase. This can be contributed to technological advances
in farming methods.

Rice quantity produced has increased over the years but seems to have
stagnated now over the past decade. California is a major producer of
rice and the on-and-off drought since 2000 has significantly decreased
rice production.

Rice producer price index has been volatile over the years with positive
YoY change, but the % YoY change is predicted to decline gradually over
the next 6 months.

Global demand for rice has been steadily rising and is expected to reach
new highs. This opens opportunities for the government to increase rice
exports and revenue. There is a growing yield with scope for more rice
production. Creating policies, subsidized loans, and financial support
that encourage farmers to grow rice and provide ease of exports can help
generate revenue.

The government should also look at water conservation and planning
policies to lessen the drought-like conditions in California, which is a
key producer of rice.
