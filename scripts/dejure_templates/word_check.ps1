# Open every template in real Microsoft Word (hidden, read-only, never saved) and
# record which ones Word cannot open. Works offline. Word must be closed first:
# the script refuses to start if Word is running, so no open document is touched.
# Uses GetType().InvokeMember: Word's type library is broken on this PC, so the
# usual $word.Documents.Open(...) style fails.
#
#   powershell -ExecutionPolicy Bypass -File word_check.ps1
#   powershell -ExecutionPolicy Bypass -File word_check.ps1 -Limit 5
param(
    [string]$Folder = "C:\Users\anton\Downloads\downloads\downloads\converted_all_final",
    [string]$Out = "C:\Users\anton\Downloads\word_check_results.csv",
    [int]$Limit = 0
)

if (Get-Process WINWORD -ErrorAction SilentlyContinue) {
    Write-Host "Word is open. Save and close your documents and Word, then run this again." -ForegroundColor Yellow
    exit 1
}

$BF = [System.Reflection.BindingFlags]
# The leading comma returns the COM object whole: PowerShell would otherwise unroll
# a collection (Documents, Paragraphs) - an empty one comes back as $null.
function Get-P($obj, $name) { return ,($obj.GetType().InvokeMember($name, $BF::GetProperty, $null, $obj, $null)) }
function Set-P($obj, $name, $val) { [void]$obj.GetType().InvokeMember($name, $BF::SetProperty, $null, $obj, @($val)) }
function Call-M($obj, $name, $argv) { return ,($obj.GetType().InvokeMember($name, $BF::InvokeMethod, $null, $obj, $argv)) }

$script:ours = @()      # PIDs of the Word instances this script started
function Start-Word {
    $before = @(Get-Process WINWORD -ErrorAction SilentlyContinue | ForEach-Object Id)
    $w = New-Object -ComObject Word.Application
    Start-Sleep -Milliseconds 500
    $script:ours += @(Get-Process WINWORD -ErrorAction SilentlyContinue | Where-Object { $before -notcontains $_.Id } | ForEach-Object Id)
    try { Set-P $w "Visible" $false; Set-P $w "DisplayAlerts" 0 } catch {}
    return $w
}
function Stop-Word($w) {
    try { [void](Call-M $w "Quit" @(0)) } catch {}
    try { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($w) } catch {}
    Start-Sleep 2
    foreach ($id in $script:ours) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
    $script:ours = @()
}

$files = Get-ChildItem -LiteralPath $Folder -Filter *.docx | Sort-Object Name
if ($Limit -gt 0) { $files = $files | Select-Object -First $Limit }
$total = @($files).Count
$results = New-Object System.Collections.Generic.List[object]
$word = Start-Word
$start = Get-Date
$i = 0
try {
    foreach ($f in $files) {
        $i++
        $t0 = Get-Date
        $status = "ok"; $paras = ""
        try {
            $docs = Get-P $word "Documents"
            # FileName, ConfirmConversions=false, ReadOnly=true, AddToRecentFiles=false
            $doc = Call-M $docs "Open" @($f.FullName, $false, $true, $false)
            $paras = Get-P (Get-P $doc "Paragraphs") "Count"
            [void](Call-M $doc "Close" @(0))   # wdDoNotSaveChanges
        } catch {
            $msg = $_.Exception.Message
            if ($_.Exception.InnerException) { $msg = $_.Exception.InnerException.Message }
            $status = "ERROR: " + $msg
            Stop-Word $word
            $word = Start-Word
        }
        $results.Add([pscustomobject]@{ file = $f.Name; status = $status; paragraphs = $paras;
                                       seconds = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1) })
        if ($i % 400 -eq 0) { Stop-Word $word; $word = Start-Word }     # keeps memory in check
        if ($i % 100 -eq 0 -or $i -eq $total) {
            $el = ((Get-Date) - $start).TotalSeconds
            Write-Host ("{0}/{1}  errors so far: {2}  (~{3:N0} min left)" -f $i, $total,
                @($results | Where-Object { $_.status -ne "ok" }).Count, ($el / $i * ($total - $i) / 60))
            $results | Export-Csv -LiteralPath $Out -NoTypeInformation -Encoding UTF8   # saved as it goes
        }
    }
} finally {
    Stop-Word $word
    $results | Export-Csv -LiteralPath $Out -NoTypeInformation -Encoding UTF8
}
$bad = @($results | Where-Object { $_.status -ne "ok" })
Write-Host ""
Write-Host ("Done: {0} files, {1} opened fine, {2} could not be opened. Results: {3}" -f $total, ($total - $bad.Count), $bad.Count, $Out)
$bad | Select-Object -First 20 | Format-Table file, status -AutoSize
