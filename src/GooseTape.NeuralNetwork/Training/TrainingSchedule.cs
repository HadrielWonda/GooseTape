namespace GooseTape.NeuralNetwork.Training;

/// <summary>
/// The hyperparameters held constant across a training run.
/// </summary>
/// <param name="LearningRate">The step size gradient descent applies.</param>
/// <param name="BatchSize">The number of examples averaged over per update.</param>
public readonly record struct TrainingSchedule(LearningRate LearningRate, BatchSize BatchSize)
{
    /// <summary>
    /// Creates a schedule from raw hyperparameter values, validating each.
    /// </summary>
    /// <param name="learningRate">The step size. Must lie within <c>(0, 1]</c>.</param>
    /// <param name="batchSize">The batch size. Must lie within <c>[1, 8192]</c>.</param>
    /// <returns>A validated <see cref="TrainingSchedule"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when either value falls outside its range.</exception>
    public static TrainingSchedule Create(double learningRate, int batchSize) =>
        new(Training.LearningRate.Create(learningRate), Training.BatchSize.Create(batchSize));
}
