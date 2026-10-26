$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$userData = [Convert]::ToBase64String([IO.File]::ReadAllBytes((Join-Path $PSScriptRoot 'user-data.sh')))
$tags = @(@{ Key = 'Project'; Value = 'VisionBridge-MLOps' }, @{ Key = 'ManagedBy'; Value = 'Codex' })
$data = @{
    ImageId = 'ami-07887b6789db08b82'
    InstanceType = 'm7i-flex.large'
    IamInstanceProfile = @{ Name = 'VisionBridgeMLOpsEC2Profile' }
    NetworkInterfaces = @(@{
        DeviceIndex = 0
        SubnetId = 'subnet-0c15cb02cbbda2e33'
        Groups = @('sg-035359fd71e75616d')
        AssociatePublicIpAddress = $true
    })
    BlockDeviceMappings = @(@{
        DeviceName = '/dev/xvda'
        Ebs = @{ VolumeSize = 30; VolumeType = 'gp3'; Encrypted = $true; DeleteOnTermination = $true }
    })
    MetadataOptions = @{ HttpTokens = 'required'; HttpEndpoint = 'enabled'; HttpPutResponseHopLimit = 2 }
    UserData = $userData
    TagSpecifications = @(
        @{ ResourceType = 'instance'; Tags = $tags },
        @{ ResourceType = 'volume'; Tags = $tags }
    )
}
$json = Join-Path $root 'mlops\aws\launch-template.generated.json'
$data | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $json -Encoding utf8
aws ec2 create-launch-template --launch-template-name visionbridge-mlops-demo --version-description 'S3 model, SSM access, no inbound ports' --launch-template-data "file://$json" --region ap-southeast-2 --profile mlops-visionbridge --query 'LaunchTemplate.LatestVersionNumber' --output text
