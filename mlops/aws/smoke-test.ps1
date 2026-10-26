$ErrorActionPreference = 'Stop'
$script = [Convert]::ToBase64String([IO.File]::ReadAllBytes((Join-Path $PSScriptRoot 'smoke-test.sh')))
$image = [Convert]::ToBase64String([IO.File]::ReadAllBytes((Join-Path $PSScriptRoot '..\..\sample\stop_sign.png')))
$command = "printf '%s' '$script' | base64 -d > /tmp/visionbridge-smoke-test.sh && printf '%s' '$image' | base64 -d > /tmp/visionbridge-stop-sign.png && bash /tmp/visionbridge-smoke-test.sh && rm -f /tmp/visionbridge-smoke-test.sh /tmp/visionbridge-stop-sign.png"
$parameters = @{ commands = @($command) } | ConvertTo-Json -Compress -Depth 5
aws ssm send-command --instance-ids i-0e44bd4d11fa9f312 --document-name AWS-RunShellScript --parameters $parameters --region ap-southeast-2 --profile mlops-visionbridge --query Command.CommandId --output text
