using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SmallExamples
{
    class Not
    {
        public static void RunTest(int SampleSize, DeviceDescriptor device)
        {
            // Create a function that converts 0 to 1 and 1 to 0
            var Input = Variable.InputVariable(new int[] { 1 }, DataType.Float, "X");
            var pointFive = new Parameter(new int[] { 1 }, DataType.Float, 0.5);
            var func = CNTKLib.Negate(CNTKLib.Minus(Input, pointFive)) + pointFive;

            // Create the input data and the expected results
            var src = new List<float>();
            Program.ExpectedResults.Clear();

            for (int i=0; i<SampleSize; i++)
            {
                if (Program.rnd.Next(2) == 0)
                {
                    src.Add(0);
                    Program.ExpectedResults.Add(1);
                } else
                {
                    src.Add(1);
                    Program.ExpectedResults.Add(0);
                }
            }

            // Run test
            var inputs = new Dictionary<Variable, Value>() { { Input, Value.CreateBatch<float>(new int[] { 1 }, src, device) } };
            var outputs = new Dictionary<Variable, Value>() { { func.Output, null } };
            func.Evaluate(inputs, outputs, device);
            Program.ActualResults = outputs[func.Output].GetDenseData<float>(func.Output).SelectMany(x => x).ToList();
        }
    }
}
