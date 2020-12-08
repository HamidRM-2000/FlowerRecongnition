using FlowerRecognition.Model;
using Microsoft.ML;
using System;

namespace FlowerRecongnition.ConsoleApp
{
   class Program
   {
      static void Main(string[] args)
      {
         //MLContext context = new MLContext(0);

         //var model=context.Model.Load("Model.zip", out _);

         //var engine = context.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

         //ModelInput sample = new ModelInput()
         //{
         //   ImagePath="..\\index.jpg"
         //};
         //var result=engine.Predict(sample);

         //Console.WriteLine("Hello World!");

         //This Line used to train the actual model
         ModelBuilder.CreateModel();
      }
   }
}
