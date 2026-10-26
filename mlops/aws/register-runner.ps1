$ErrorActionPreference = 'Stop'
$credentialLines = "protocol=https`nhost=github.com`n`n" | git credential fill
$tokenLine = $credentialLines | Where-Object { $_ -like 'password=*' } | Select-Object -First 1
if (-not $tokenLine) { throw 'GitHub credential is unavailable' }
$authToken = $tokenLine.Substring(9)
$headers = @{
    Authorization = "Bearer $authToken"
    Accept = 'application/vnd.github+json'
    'X-GitHub-Api-Version' = '2022-11-28'
}
$registration = Invoke-RestMethod -Method Post -Uri 'https://api.github.com/repos/Vkmstry1611/mai-project/actions/runners/registration-token' -Headers $headers
$runnerToken = [string]$registration.token
if ($runnerToken -notmatch '^[A-Za-z0-9]+$') { throw 'Unexpected runner token format' }
$script = [Convert]::ToBase64String([IO.File]::ReadAllBytes((Join-Path $PSScriptRoot 'install-runner.sh')))
$command = "printf '%s' '$script' | base64 -d > /tmp/visionbridge-install-runner.sh && RUNNER_TOKEN='$runnerToken' bash /tmp/visionbridge-install-runner.sh && rm -f /tmp/visionbridge-install-runner.sh"
$parameters = @{ commands = @($command) } | ConvertTo-Json -Compress -Depth 5
aws ssm send-command --instance-ids i-0e44bd4d11fa9f312 --document-name AWS-RunShellScript --parameters $parameters --region ap-southeast-2 --profile mlops-visionbridge --query Command.CommandId --output text
