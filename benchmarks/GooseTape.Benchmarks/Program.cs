using BenchmarkDotNet.Running;
using GooseTape.Benchmarks;

// Run every benchmark:            dotnet run -c Release --project benchmarks/GooseTape.Benchmarks
// Run one of them:                dotnet run -c Release --project benchmarks/GooseTape.Benchmarks -- --filter *Matrix*
// Results are written to BenchmarkDotNet.Artifacts/, which is git-ignored.
BenchmarkSwitcher.FromAssembly(typeof(MatrixBenchmarks).Assembly).Run(args);
