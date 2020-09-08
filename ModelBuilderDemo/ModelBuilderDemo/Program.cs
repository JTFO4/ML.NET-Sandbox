using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

namespace ForecastDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            // GET root directory
            string rootDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "../../../"));

            // INIT input and output paths
            string dbFilePath = Path.Combine(rootDir, "Data", "DailyDemand.mdf");
            string modelPath = Path.Combine(rootDir, "MLModel.zip");

            // INIT connection string
            var connectionString = $"Data Source=(LocalDB)\\MSSQLLocalDB;AttachDbFilename={dbFilePath};Integrated Security=True;Connect Timeout=30;";

            // INIT ML context
            MLContext mlContext = new MLContext();

            // INIT loader to pull data from DB
            DatabaseLoader loader = mlContext.Data.CreateDatabaseLoader<ModelInput>();

            // INIT query
            string query = "SELECT RentalDate, CAST(Year as REAL) as Year, CAST(TotalRentals as REAL) as TotalRentals FROM Rentals";

            // INIT dbSource to connect to the DB and execute the query
            DatabaseSource dbSource = new DatabaseSource(SqlClientFactory.Instance,
                                connectionString,
                                query);

            // INIT dataView to load the DB data into
            IDataView dataView = loader.Load(dbSource);

            // SPLIT train/test
            IDataView firstYearData = mlContext.Data.FilterRowsByColumn(dataView, "Year", upperBound: 1);
            IDataView secondYearData = mlContext.Data.FilterRowsByColumn(dataView, "Year", lowerBound: 1);

            // INIT ML time series analysis pipeline
            var forecastingPipeline = mlContext.Forecasting.ForecastBySsa(
                                                            outputColumnName: "ForecastedRentals",
                                                            inputColumnName: "TotalRentals",
                                                            windowSize: 7,
                                                            seriesLength: 30,
                                                            trainSize: 365,
                                                            horizon: 7,
                                                            confidenceLevel: 0.95f,
                                                            confidenceLowerBoundColumn: "LowerBoundRentals",
                                                            confidenceUpperBoundColumn: "UpperBoundRentals");

            // FIT the model to the training set
            SsaForecastingTransformer forecaster = forecastingPipeline.Fit(firstYearData);

            // EVAL the model with the test set
            Evaluate(secondYearData, forecaster, mlContext);

            // SAVE the model
            var forecastEngine = forecaster.CreateTimeSeriesEngine<ModelInput, ModelOutput>(mlContext);
            forecastEngine.CheckPoint(mlContext, modelPath);

            //  RUN the application
            Forecast(secondYearData, 7, forecastEngine, mlContext);

        }

        static void Evaluate(IDataView testData, ITransformer model, MLContext mlContext)
        {
            // INIT predictions
            IDataView predictions = model.Transform(testData);

            // GET expected output data
            IEnumerable<float> actual =
                mlContext.Data.CreateEnumerable<ModelInput>(testData, true)
                    .Select(observed => observed.TotalRentals);

            // GET predicted output data
            IEnumerable<float> forecast =
                mlContext.Data.CreateEnumerable<ModelOutput>(predictions, true)
                    .Select(prediction => prediction.ForecastedRentals[0]);

            // GET the prediction error
            var metrics = actual.Zip(forecast, (actualValue, forecastValue) => actualValue - forecastValue);

            // GET MAE and RMSE metric data
            var MAE = metrics.Average(error => Math.Abs(error)); // Mean Absolute Error
            var RMSE = Math.Sqrt(metrics.Average(error => Math.Pow(error, 2))); // Root Mean Squared Error

            Console.WriteLine("Evaluation Metrics");
            Console.WriteLine("---------------------");
            Console.WriteLine($"Mean Absolute Error: {MAE:F3}");
            Console.WriteLine($"Root Mean Squared Error: {RMSE:F3}\n");
        }

        static void Forecast(IDataView testData, int horizon, TimeSeriesPredictionEngine<ModelInput, ModelOutput> forecaster, MLContext mlContext)
        {
            // GET single prediction
            ModelOutput forecast = forecaster.Predict();

            // INIT prediction output
            IEnumerable<string> forecastOutput =
                mlContext.Data.CreateEnumerable<ModelInput>(testData, reuseRowObject: false)
                    .Take(horizon)
                    .Select((ModelInput rental, int index) =>
                    {
                        string rentalDate = rental.RentalDate.ToShortDateString();
                        float actualRentals = rental.TotalRentals;
                        float lowerEstimate = Math.Max(0, forecast.LowerBoundRentals[index]);
                        float estimate = forecast.ForecastedRentals[index];
                        float upperEstimate = forecast.UpperBoundRentals[index];
                        return $"Date: {rentalDate}\n" +
                        $"Actual Rentals: {actualRentals}\n" +
                        $"Lower Estimate: {lowerEstimate}\n" +
                        $"Forecast: {estimate}\n" +
                        $"Upper Estimate: {upperEstimate}\n";
                    });

            Console.WriteLine("Rental Forecast");
            Console.WriteLine("---------------------");
            foreach (var prediction in forecastOutput)
            {
                Console.WriteLine(prediction);
            }
        }
    }

    public class ModelInput
    {
        public DateTime RentalDate { get; set; }

        public float Year { get; set; }

        public float TotalRentals { get; set; }
    }

    public class ModelOutput
    {
        public float[] ForecastedRentals { get; set; }

        public float[] LowerBoundRentals { get; set; }

        public float[] UpperBoundRentals { get; set; }
    }
}
