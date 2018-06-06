using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ComputationalExamples
{
    class And
    {
        public static void RunTest(int SampleSize, DeviceDescriptor device)
        {
            // Create a function that performs a logical AND operation. 0 is false and 1 is true
            var X = Variable.InputVariable(new int[] { 1 }, DataType.Float, "X");
            var Y = Variable.InputVariable(new int[] { 1 }, DataType.Float, "Y");

            var one = new Parameter(new int[] { 1 }, DataType.Float, 1);

            var func = CNTKLib.ReLU(CNTKLib.Minus(X + Y, one));

            // Create the input data and the expected results
            var x = new List<float>();
            var y = new List<float>();

            Program.ExpectedResults.Clear();
            var rnd = Program.rnd;
            for (int i = 0; i < SampleSize; i++)
            {
                x.Add(rnd.Next(2));
                y.Add(rnd.Next(2));
                Program.ExpectedResults.Add(((x.Last() == 1) && (y.Last() == 1)) ? 1 : 0);
            }

            // Run test
            var inputs = new Dictionary<Variable, Value>()
            {
                { X, Value.CreateBatch(new int[] { 1 }, x, device) },
                { Y, Value.CreateBatch(new int[] { 1 }, y, device) }
            };
            var outputs = new Dictionary<Variable, Value>() { { func.Output, null } };
            func.Evaluate(inputs, outputs, device);
            Program.ActualResults = outputs[func.Output].GetDenseData<float>(func.Output).SelectMany(item => item).ToList();
        }
    }
}
