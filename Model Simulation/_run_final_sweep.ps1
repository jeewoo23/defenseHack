# Final sweep: 4 scenarios x 4 controllers x 20 seeds
# Includes cooperative observation (cap 2) and reduced strike window (12 steps).
$scenarios = @(
    @{ Label = "outnumbered"; Uavs = 10; Enemies = 5; Flank = 5 },
    @{ Label = "even";        Uavs = 12; Enemies = 4; Flank = 4 },
    @{ Label = "current";     Uavs = 14; Enemies = 3; Flank = 3 },
    @{ Label = "abundant";    Uavs = 18; Enemies = 2; Flank = 2 }
)
$py = "C:\Users\jeewo\Documents\defenseHack\Model Simulation\.venv\Scripts\python.exe"
$seeds = 20
$workers = 10

foreach ($s in $scenarios) {
    $args_common = @(
        "benchmark.py",
        "--scenario", "dual_objective",
        "--seeds", $seeds,
        "--workers", $workers,
        "--n-uavs", $s.Uavs,
        "--n-enemies", $s.Enemies,
        "--n-flank", $s.Flank
    )
    foreach ($ctrl in @("greedy","intercept","defense","horizon")) {
        $prefix = "benchmark_final_$($s.Label)_${ctrl}_20seed"
        $ctrl_args = switch ($ctrl) {
            "greedy"    { @("--policy","greedy","--emergency-intercept","off","--objective-defense","off") }
            "intercept" { @("--policy","greedy","--emergency-intercept","on","--objective-defense","off") }
            "defense"   { @("--policy","greedy","--emergency-intercept","off","--objective-defense","on") }
            "horizon"   { @("--policy","horizon") }
        }
        $cli = $args_common + $ctrl_args + @("--output-prefix", $prefix)
        Write-Output "=== $($s.Label) / $ctrl ==="
        $t0 = Get-Date
        & $py $cli 2>&1 | Select-Object -Last 2
        $dt = ((Get-Date) - $t0).TotalSeconds
        Write-Output "  done in $([math]::Round($dt,1))s"
    }
}
Write-Output "FINAL SWEEP COMPLETE"
