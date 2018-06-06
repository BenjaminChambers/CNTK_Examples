using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace ComputationalExamples
{
    public class mAdd
    {
        public static void RunTest(int SampleSize, DeviceDescriptor device)
        {
            var single = new int[] { 1 };

            // Create a function that performs a multiply-add on 3 inputs and returns the result
            var X = Variable.InputVariable(single, DataType.Float, "X");
            var Y = Variable.InputVariable(single, DataType.Float, "Y");
            var Z = Variable.InputVariable(single, DataType.Float, "Z");

            var func = (X * Y) + Z;

            // Create the input data and the expected results
            var x = new List<float>();
            var y = new List<float>();
            var z = new List<float>();

            Program.ExpectedResults.Clear();
            var rnd = Program.rnd;
            for (int i = 0; i < SampleSize; i++)
            {
                x.Add((float)rnd.NextDouble());
                y.Add((float)rnd.NextDouble());
                z.Add((float)rnd.NextDouble());

                Program.ExpectedResults.Add((x.Last()*y.Last())+z.Last());
            }

            // Run test
            var inputs = new Dictionary<Variable, Value>()
            {
                { X, Value.CreateBatch(single, x, device) },
                { Y, Value.CreateBatch(single, y, device) },
                { Z, Value.CreateBatch(single, z, device) }
            };
            var outputs = new Dictionary<Variable, Value>() { { func.Output, null } };
            func.Evaluate(inputs, outputs, device);
            Program.ActualResults = outputs[func.Output].GetDenseData<float>(func.Output).SelectMany(item => item).ToList();
        }
    }
}
