using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace ActivationFunctions
{
    // A small program demonstrating the effect of various activation functions
    class Program
    {
        static DeviceDescriptor device = DeviceDescriptor.CPUDevice;
        static List<float> raw = new List<float>();
        static Variable inVar;
        static Value inVal;

        static void Test(string Name, Function func)
        {
            Console.WriteLine("Testing function {0}", Name);
            Console.Write("Inputs:");
            foreach (var item in raw)
                Console.Write("\t{0:0.000}", item);
            Console.WriteLine();
            Console.Write("Output:");

            var inputs = new Dictionary<Variable, Value>() { { inVar, inVal } };
            var outputs = new Dictionary<Variable, Value>() { { func.Output, null } };
            func.Evaluate(inputs, outputs, device);
            var results = outputs[func.Output].GetDenseData<float>(func.Output)[0];
            foreach (var item in results)
                Console.Write("\t{0:0.000}", item);
            Console.WriteLine();
            Console.WriteLine();
        }

        static void Main(string[] args)
        {
            for (float i = -1.0f; i <= 1.0f; i += 0.1f)
                raw.Add(i);
            var shape = new int[] { raw.Count };

            inVar = Variable.InputVariable(shape, DataType.Float);
            inVal = Value.CreateBatch(shape, raw, device);

            var relu = CNTKLib.ReLU(inVar);
            var sigm = CNTKLib.Sigmoid(inVar);
            var tanh = CNTKLib.Tanh(inVar);
            var softm = CNTKLib.Softmax(inVar);
            var lsoft = CNTKLib.LogSoftmax(inVar);
            var hardm = CNTKLib.Hardmax(inVar);

            Test("ReLU", relu);
            Test("Sigmoid", sigm);
            Test("Tanh", tanh);
            Test("Softmax", softm);
            Test("Log Softmax", lsoft);
            Test("Hardmax", hardm);
        }
    }
}
