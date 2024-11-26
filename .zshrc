# Amazon Q blocks
[[ -f "${HOME}/Library/Application Support/amazon-q/shell/zshrc.pre.zsh" ]] && builtin source "${HOME}/Library/Application Support/amazon-q/shell/zshrc.pre.zsh"

# Load environment variables
if [[ -f "$HOME/git/BACKUP_ENV/.env" ]]; then
    set -o allexport
    source "$HOME/git/BACKUP_ENV/.env"
    set +o allexport
fi

# Path configuration
typeset -U path  # Ensure unique PATH entries
path=(
    "$HOME/Library/Python/3.12/bin"  # Add Python user bin directory
    /usr/local/opt/python@3.12/bin
    /usr/local/bin
    /usr/local/sbin
    $path
)

# Python configuration
export PYTHONPATH="$HOME/miniconda3/lib/python3.12/site-packages:$PYTHONPATH"

# Fix pip dependency conflicts
export PIP_REQUIRE_VIRTUALENV=false  # Allow global pip installations
export PIP_BREAK_SYSTEM_PACKAGES=1   # Allow pip to manage system packages

# Conda initialization
__conda_setup="$('/opt/anaconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    for conda_path in "$HOME/miniconda3" "/opt/anaconda3"; do
        if [ -f "$conda_path/etc/profile.d/conda.sh" ]; then
            . "$conda_path/etc/profile.d/conda.sh"
            break
        fi
    done
fi
unset __conda_setup

# Automatically activate conda base environment
conda activate base 2>/dev/null

# Aliases
alias {
    bac="bash -n ~/git/BACKUP_ENV/.env"
    backup="backup -v -d /Users/samihalawa/"
    backup-system="sudo system_profiler SPTrashDataType | grep '(Total [0-9]+G)'"
    clean='find /var/log -type f -name "*.log" | xargs rm -rfv "$(hostname)"'
    st='streamlit run'
    python="python3.12"
    pip="pip3.12"
    fix-deps='pip install --upgrade pip && pip install --upgrade setuptools wheel'
}

# Secrets and tokens
export GITSYNC_PASSWORD='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'

# Amazon Q post block
[[ -f "${HOME}/Library/Application Support/amazon-q/shell/zshrc.post.zsh" ]] && builtin source "${HOME}/Library/Application Support/amazon-q/shell/zshrc.post.zsh" 