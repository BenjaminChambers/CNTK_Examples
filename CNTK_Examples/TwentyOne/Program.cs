using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace TwentyOne
{
    class Program
    {
        static Random rnd = new Random();

        static int DrawCard()
        {
            // As the point of this exercise is to train a NN and not to write a full card game, we'll be using a simplified deck
            // Only one suit, aces are low, and every card has an equal chance of being dealt (regardless of deck state)

            var card = rnd.Next(13) + 1;
            return (card < 10) ? card : 10;
        }

        static float[] HandToArray(int Hand)
        {
            var data = new float[20];
            data[Hand-1] = 1;
            return data;
        }

        static Function BuildFunction(DeviceDescriptor device)
        {
            var Hand = Variable.InputVariable(new int[] { 20 }, DataType.Float, "Hand");

            int hidden = 50;
            var iWeights = new Parameter(new int[] { hidden, 20 }, DataType.Float, CNTKLib.GlorotNormalInitializer());
            var iBias = new Parameter(new int[] { hidden }, DataType.Float, 0);
            var oWeights = new Parameter(new int[] { 2, hidden }, DataType.Float, CNTKLib.GlorotNormalInitializer());
            var oBias = new Parameter(new int[] { 2 }, DataType.Float, 0);

            var layer = CNTKLib.ReLU((iWeights * Hand) + iBias);
            return (oWeights * layer) + oBias;
        }

        static (int Hand, bool HitMe, bool ShouldHit) PlayHand(Function func, int hand, DeviceDescriptor device)
        {
            var shouldHit = (hand + DrawCard()) <= 21;

            var inVar = func.Inputs.First(x => x.Name == "Hand");
            var inputs = new Dictionary<Variable, Value>() { { inVar, Value.CreateBatch(new int[] { 20 }, HandToArray(hand), device) } };
            var outputs = new Dictionary<Variable, Value>() { { func.Output, null } };
            func.Evaluate(inputs, outputs, device);
            var result = outputs[func.Output].GetDenseData<float>(func.Output).First();

            var hitMe = result[1] > result[0];

            return (hand, hitMe, shouldHit);
        }

        static void TrainPlayer(Function func, IEnumerable<(int Hand, bool, bool ShouldHit)> Data, DeviceDescriptor device)
        {
            var rawFeatures = new List<float>();
            var rawLabels = new List<float>();

            foreach (var item in Data)
            {
                rawFeatures.AddRange(HandToArray(item.Hand));
                rawLabels.AddRange(item.ShouldHit ? new float[] { 0, 1 } : new float[] { 1, 0 });
            }

            var features = Value.CreateBatch(new int[] { 20 }, rawFeatures, device);
            var labels = Value.CreateBatch(new int[] { 2 }, rawLabels, device);

            var labelVar = Variable.InputVariable(new int[] { 2 }, DataType.Float);

            var loss = CNTKLib.CrossEntropyWithSoftmax(func, labelVar);
            var evalError = CNTKLib.ClassificationError(func, labelVar);
            var learningRatePerSample = new TrainingParameterScheduleDouble(0.001, 1);
            var parameterLearners = new List<Learner>() { Learner.SGDLearner(func.Parameters(), learningRatePerSample) };
            var trainer = Trainer.CreateTrainer(func, loss, evalError, parameterLearners);

            var trainingData = new Dictionary<Variable, Value>() { { func.Inputs.First(x=>x.Name=="Hand"), features }, { labelVar, labels } };

            for (int i=0; i<1000; i++)
                trainer.TrainMinibatch(trainingData, false, device);
        }

        static void Main(string[] args)
        {
            var device = DeviceDescriptor.CPUDevice;

            Console.WriteLine("This is a small program to demonstrate training a Neural Network.");
            Console.WriteLine("The NN will learn whether to hit or hold in a simplified game of BlackJack.");

            Console.WriteLine("Creating the model.");
            var player = BuildFunction(device);

            Console.WriteLine("Playing one hundred hands...");
            var hands = new List<(int Hand, bool HitMe, bool ShouldHit)>();
            for (int i = 0; i < 100; i++)
                hands.Add(PlayHand(player, DrawCard() + DrawCard(), device));

            int playedWell = hands.Where(x => x.HitMe == x.ShouldHit).Count();
            Console.WriteLine("Optimal move made {0} times.", playedWell);

            Console.WriteLine("Learning from those games; training for 100 epochs...");

            TrainPlayer(player, hands, device);

            Console.WriteLine("Playing another hundred hands...");
            var secondSet = new List<(int Hand, bool HitMe, bool ShouldHit)>();
            for (int i = 0; i < 100; i++)
                secondSet.Add(PlayHand(player, DrawCard() + DrawCard(), device));

            playedWell = secondSet.Where(x => x.HitMe == x.ShouldHit).Count();
            Console.WriteLine("Optimal move made {0} times.", playedWell);
        }
    }
}
