using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FlowerRecognition.Model
{
   public class ModelOutput
   {
      public string PredictedLabel { get; set; }
      public float[] Score { get; set; }
   }
}
