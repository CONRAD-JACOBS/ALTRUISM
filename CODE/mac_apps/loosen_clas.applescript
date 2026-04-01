set reposRoot to "/Users/socialneurolab/Documents/repos"

tell application "Terminal"
    activate
    do script "clear; echo '=== LOOSEN CLAS ==='; cd " & quoted form of (reposRoot & "/uq-neuro-nao") & " && /Library/Frameworks/Python.framework/Versions/2.7/bin/python -m src_py2.main.loosen_clas; echo ''; echo 'Press any key to close this window.'; read -n 1"
end tell
