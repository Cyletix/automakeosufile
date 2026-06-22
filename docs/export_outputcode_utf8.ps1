$root = "D:\OneDrive\Project\AutoMakeosuFile"
$output = Join-Path $root "docs\outputcode.md"
$targets = @(
    (Join-Path $root "main.py")
) + (
    Get-ChildItem -Path (Join-Path $root "automakeosufile") -Recurse -File -Filter *.py |
        Where-Object { $_.FullName -notmatch '\\__pycache__\\' } |
        Sort-Object FullName |
        Select-Object -ExpandProperty FullName
)

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$utf8Bom = New-Object System.Text.UTF8Encoding($true)
$sb = New-Object System.Text.StringBuilder

foreach ($full in $targets) {
    $relative = $full.Substring($root.Length + 1).Replace('\', '/')
    [void]$sb.AppendLine($relative)
    [void]$sb.AppendLine('```python')

    $content = [System.IO.File]::ReadAllText($full, $utf8NoBom)
    if ($null -eq $content) { $content = '' }

    [void]$sb.Append($content)
    if (-not $content.EndsWith("`n")) {
        [void]$sb.AppendLine()
    }

    [void]$sb.AppendLine('```')
    [void]$sb.AppendLine()
    [void]$sb.AppendLine('---')
    [void]$sb.AppendLine()
}

[System.IO.File]::WriteAllText($output, $sb.ToString(), $utf8Bom)
Write-Output "Saved: $output"