using FlowerRecognition.Model;
using Microsoft.ML;
using Microsoft.ML.Vision;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace FlowerRecongnition.ConsoleApp
{
   public static class ModelBuilder
   {
      private static string DATA_PATH = Path.Combine("..", "..", "..", "flowers");
      private static string WORK_SPACE_PATH = Path.Combine("..", "..", "..", "flowers");
      public static void CreateModel()
      {
         MLContext context = new MLContext(0);

         //Convert and Load data
         IEnumerable<ModelInput> convertedData = ConvertData();
         IDataView dataView = context.Data.LoadFromEnumerable(convertedData);
         dataView = context.Data.ShuffleRows(dataView);


         //Split Data 
         var splitedData = context.Data.TrainTestSplit(dataView, .3);

         //Build PipeLine
         IEstimator<ITransformer> trainingPipeLine = BuildPipeLine(context);

         //Train Model
         ITransformer model = TrainModel(trainingPipeLine, splitedData.TrainSet);

         //Evaluate Model
         EvaluateModel(context, model, splitedData.TestSet);

         //Save the Model
         SaveModel(context, model);
      }


      private static void SaveModel(MLContext context, ITransformer model)
      {
         Console.WriteLine($"------------- Saving the Model -------------");
         context.Model.Save(model, null, "Model.zip");
      }

      private static void EvaluateModel(MLContext context, ITransformer model, IDataView testSet)
      {
         Console.WriteLine("------------- Evaluating Model Metrics -------------");
         var transfomedData = model.Transform(testSet);

         var result = context.MulticlassClassification.Evaluate(transfomedData);

         Console.WriteLine($" {result.LogLoss}");
         Console.WriteLine($" {result.LogLossReduction}");
         Console.WriteLine($" {result.MacroAccuracy}");
         Console.WriteLine($" {result.MicroAccuracy}");
         Console.WriteLine($" {result.TopKAccuracy}");
         Console.WriteLine($" {result.TopKPredictionCount}");

         Console.WriteLine("------------- ------------- ------------- -------------");
      }

      private static ITransformer TrainModel(IEstimator<ITransformer> trainingPipeLine, IDataView trainSet)
      {
         Console.WriteLine("------------- Start Training -------------");
         var model = trainingPipeLine.Fit(trainSet);
         Console.WriteLine("------------- Training Finished -------------");
         Console.WriteLine("------------- ------------- ------------- -------------");
         return model;
      }

      private static IEstimator<ITransformer> BuildPipeLine(MLContext context)
      {
         var dataPipeLine = context.Transforms.Conversion.MapValueToKey("Label", "Label")
            .Append(context.Transforms.LoadRawImageBytes("Image", DATA_PATH, "ImagePath"));
         var trainerOptions = new ImageClassificationTrainer.Options()
         {
            LabelColumnName = "Label",
            FeatureColumnName = "Image",
            Arch = ImageClassificationTrainer.Architecture.ResnetV250,
            BatchSize = 20,
            MetricsCallback = Console.WriteLine 
         };

         var trainer = context.MulticlassClassification.Trainers.ImageClassification(trainerOptions);

         var trainingPipeLine = dataPipeLine.Append(trainer)
            .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
         return trainingPipeLine;
      }

      private static IEnumerable<ModelInput> ConvertData()
      {
         List<ModelInput> images = new List<ModelInput>();
         var imageFiles = Directory.GetFiles(DATA_PATH, "*", SearchOption.AllDirectories);
         foreach (var item in imageFiles)
         {
            images.Add(new ModelInput()
            {
               ImagePath = Path.GetFullPath(item),
               Label = Directory.GetParent(item).Name
            });
         }
         return images;
      }
   }
}
