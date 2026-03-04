#!/usr/bin/env python3
"""
Script d'installation automatique TensorFlow GPU
=================================================
Détecte votre système et installe TensorFlow avec support GPU.
Basé sur le script install_tensorflow_gpu avec ajouts uv et améliorations de la détection.

Utilisation : python install_tensorflow_gpu_v2.py

Auteur : Jeremy K.
Date   : Mars 2026
"""

import os
import sys
import platform
import subprocess
import shutil


# =============================================================================
# CONFIGURATIONS PAR PLATEFORME
# Toutes les variations OS sont ici — pas dans la logique d'installation.
# =============================================================================

PLATFORM_CONFIGS = {
    "linux": {
        "label":      "Linux / Ubuntu",
        "tf_package": "tensorflow[and-cuda]",
        "python":     "3.11",
        "requires_nvidia": True,
    },
    "wsl": {
        "label":      "Windows WSL2",
        "tf_package": "tensorflow[and-cuda]",
        "python":     "3.11",
        "requires_nvidia": True,
        "nvidia_note": (
            "⚠️  Sur WSL2, les drivers NVIDIA doivent être installés côté Windows.\n"
            "   NE PAS installer CUDA directement dans WSL2."
        ),
    },
    "macos_arm": {
        "label":      "macOS Apple Silicon (M1/M2/M3/M4)",
        "tf_package": "tensorflow-macos tensorflow-metal",
        "python":     "3.11",
        "requires_nvidia": False,
    },
    "macos_x86": {
        "label":      "macOS Intel",
        "tf_package": "tensorflow",
        "python":     "3.11",
        "requires_nvidia": False,
        "gpu_note": "ℹ️  Mac Intel : support GPU non disponible, installation CPU uniquement.",
    },
    "windows_native": {
        "label":      "Windows natif",
        "tf_package": "tensorflow<2.11",
        "python":     "3.10",
        "requires_nvidia": False,
        "requires_conda": True,
        "conda_cuda":  "cudatoolkit=11.2 cudnn=8.1.0",
        "warning": (
            "⚠️  ATTENTION : le support GPU sur Windows natif est limité à TensorFlow ≤ 2.10.\n"
            "   Recommandation : utilisez WSL2 pour les versions récentes."
        ),
    },
}


# =============================================================================
# UTILITAIRES
# =============================================================================

def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_step(num: int, description: str):
    print(f"\n📋 Étape {num} : {description}")


def ask_confirm(prompt: str) -> bool:
    return input(f"   {prompt} (y/n) : ").strip().lower() == "y"


def command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run(cmd: str, description: str = "", check: bool = True) -> bool:
    """Affiche une commande, demande confirmation, puis l'exécute."""
    if description:
        print(f"\n   → {description}")
    print(f"   $ {cmd}")

    if not ask_confirm("Exécuter cette commande ?"):
        print("   ⏭️  Étape ignorée.")
        return False

    try:
        subprocess.run(cmd, shell=True, check=check)
        print("   ✅ Succès.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Échec (code {e.returncode}).")
        return False


# =============================================================================
# DÉTECTION DU SYSTÈME
# =============================================================================

def detect_system() -> dict:
    os_name    = platform.system()
    arch       = platform.machine()
    py_version = platform.python_version()

    # WSL : la release contient "Microsoft" ou "microsoft"
    is_wsl = (
        os_name == "Linux"
        and "microsoft" in platform.release().lower()
    )

    return {
        "os":          os_name,
        "arch":        arch,
        "py_version":  py_version,
        "is_wsl":      is_wsl,
        "has_conda":   command_exists("conda"),
        "has_uv":      command_exists("uv"),
        "has_nvidia":  command_exists("nvidia-smi"),
        "in_venv":     _in_virtual_env(),
        "active_env":  _active_env_name(),
    }


def _in_virtual_env() -> bool:
    """Retourne True si un environnement virtuel est déjà actif."""
    in_venv  = sys.prefix != sys.base_prefix
    in_conda = os.environ.get("CONDA_DEFAULT_ENV") not in (None, "base")
    return in_venv or in_conda


def _active_env_name() -> str:
    """Retourne le nom de l'environnement actif, ou une chaîne vide."""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env and conda_env != "base":
        return f"conda:{conda_env}"
    venv_path = os.environ.get("VIRTUAL_ENV", "")
    if venv_path:
        return f"venv:{os.path.basename(venv_path)}"
    return ""


# =============================================================================
# RÉSOLUTION DE LA PLATEFORME
# =============================================================================

def resolve_platform(info: dict) -> str | None:
    """
    Détermine la clé de config à utiliser selon l'OS détecté.
    Retourne None si la plateforme n'est pas supportée.
    """
    os_name = info["os"]

    if os_name == "Linux":
        return "wsl" if info["is_wsl"] else "linux"

    if os_name == "Darwin":
        return "macos_arm" if info["arch"].startswith("arm") else "macos_x86"

    if os_name == "Windows":
        print("\n🪟  Windows détecté.")
        print("   Deux méthodes d'installation sont disponibles :")
        print("   1. WSL2 (recommandé) — supporte les versions récentes de TensorFlow")
        print("   2. Windows natif     — limité à TensorFlow ≤ 2.10")

        choice = input("\n   Votre choix (1/2) : ").strip()
        if choice == "1":
            print(
                "\n   Pour utiliser WSL2, lancez Ubuntu depuis le menu Démarrer\n"
                "   puis ré-exécutez ce script à l'intérieur.\n"
                "   Installation WSL2 (PowerShell admin) : wsl --install -d Ubuntu-24.04"
            )
            return None
        return "windows_native"

    return None


# =============================================================================
# CRÉATION DE L'ENVIRONNEMENT VIRTUEL
# =============================================================================

def create_env(info: dict, config: dict) -> bool:
    """
    Crée un environnement virtuel avec l'outil disponible.
    Ordre de priorité : env déjà actif > conda > uv > venv.
    Retourne False si l'utilisateur annule ou si une erreur bloquante survient.
    """

    # --- Env déjà actif : pas besoin d'en créer un nouveau ---
    if info["in_venv"]:
        print(f"\n✅ Environnement virtuel déjà actif : {info['active_env']}")
        print("   Installation directement dans cet environnement.")
        return True

    python_ver = config["python"]
    env_name   = "tf-gpu"

    print(f"\n   Aucun environnement actif détecté.")
    print(f"   Création d'un environnement '{env_name}' (Python {python_ver})...")

    # --- Conda ---
    if info["has_conda"]:
        print("\n🐍 Conda détecté.")

        # Windows natif avec CUDA via conda
        if config.get("requires_conda") and config.get("conda_cuda"):
            ok = run(
                f"conda create --name {env_name} python={python_ver} -y",
                f"Création de l'environnement conda '{env_name}'"
            )
            if not ok:
                return False
            run(
                f"conda install -c conda-forge {config['conda_cuda']} -y",
                "Installation de CUDA et cuDNN via conda"
            )
        else:
            run(
                f"conda create --name {env_name} python={python_ver} -y",
                f"Création de l'environnement conda '{env_name}'"
            )

        print(f"\n⚠️  Activez l'environnement avant la suite :")
        print(f"   conda activate {env_name}")
        print("   Puis relancez ce script.")
        return False   # L'utilisateur doit activer et relancer

    # --- uv ---
    if info["has_uv"]:
        print("\n⚡ uv détecté.")
        ok = run(
            f"uv venv {env_name} --python {python_ver}",
            f"Création de l'environnement uv '{env_name}'"
        )
        if not ok:
            return False
        print(f"\n⚠️  Activez l'environnement avant la suite :")
        _print_activate_cmd(env_name, info["os"])
        print("   Puis relancez ce script.")
        return False   # L'utilisateur doit activer et relancer

    # --- venv standard ---
    print("\n🐍 Utilisation de venv (standard).")
    ok = run(
        f"python{python_ver} -m venv {env_name}",
        f"Création de l'environnement venv '{env_name}'"
    )
    if not ok:
        return False
    print(f"\n⚠️  Activez l'environnement avant la suite :")
    _print_activate_cmd(env_name, info["os"])
    print("   Puis relancez ce script.")
    return False   # L'utilisateur doit activer et relancer


def _print_activate_cmd(env_name: str, os_name: str):
    if os_name == "Windows":
        print(f"   {env_name}\\Scripts\\activate")
    else:
        print(f"   source {env_name}/bin/activate")


# =============================================================================
# VÉRIFICATIONS PRÉALABLES
# =============================================================================

def check_prerequisites(info: dict, config: dict) -> bool:
    """Vérifie que tous les prérequis sont remplis avant l'installation."""

    # --- Version Python ---
    parts = tuple(map(int, info["py_version"].split(".")[:2]))
    if not (3, 9) <= parts <= (3, 12):
        print(f"\n❌ Python {info['py_version']} non supporté.")
        print("   TensorFlow 2.17+ requiert Python 3.9 – 3.12.")
        return False

    # --- GPU NVIDIA (si requis) ---
    if config.get("requires_nvidia"):
        if config.get("nvidia_note"):
            print(f"\n{config['nvidia_note']}")

        if not info["has_nvidia"]:
            print("\n❌ nvidia-smi introuvable — GPU NVIDIA non détecté.")
            _print_nvidia_install_help(info)
            return False

        print("\n✅ GPU NVIDIA détecté (nvidia-smi disponible).")

    # --- Conda requis (Windows natif) ---
    if config.get("requires_conda") and not info["has_conda"]:
        print("\n❌ Conda est requis pour l'installation Windows natif.")
        print("   Installez Miniconda : https://docs.conda.io/en/latest/miniconda.html")
        return False

    # --- Note GPU absente (macOS Intel, etc.) ---
    if config.get("gpu_note"):
        print(f"\n{config['gpu_note']}")

    # --- Avertissement plateforme (Windows natif) ---
    if config.get("warning"):
        print(f"\n{config['warning']}")
        if not ask_confirm("Continuer malgré tout ?"):
            return False

    return True


def _print_nvidia_install_help(info: dict):
    if info["is_wsl"]:
        print("   → Installez les drivers NVIDIA pour Windows depuis :")
        print("     https://www.nvidia.com/Download/index.aspx")
        print("   → Ne pas installer CUDA séparément dans WSL2.")
    else:
        print("   → Installez les drivers NVIDIA :")
        print("     sudo apt update")
        print("     sudo apt install nvidia-driver-535")
        print("     sudo reboot")


# =============================================================================
# INSTALLATION DES PACKAGES
# =============================================================================

def install_packages(info: dict, config: dict) -> bool:
    """Installe TensorFlow (et les éventuelles dépendances Metal sur macOS)."""

    pip_cmd = "uv pip" if info["has_uv"] and info["in_venv"] else "pip"

    run(f"{pip_cmd} install --upgrade pip", "Mise à jour de pip")

    # Plusieurs packages possibles (ex: tensorflow-macos + tensorflow-metal)
    for package in config["tf_package"].split():
        ok = run(f"{pip_cmd} install '{package}'", f"Installation de {package}")
        if not ok:
            print(f"\n❌ L'installation de '{package}' a échoué ou a été annulée.")
            return False

    return True


# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================

def print_success_summary():
    print_header("INSTALLATION TERMINÉE")
    print("🎉 TensorFlow a été installé avec succès !\n")
    print("📋 Prochaines étapes :")
    print("   1. Vérifiez que votre environnement virtuel est bien activé.")
    print("   2. Testez l'installation avec les commandes ci-dessous.")
    print("   3. Consultez le guide complet : tensorflow-gpu-setup.md\n")
    print("🔍 Commandes de vérification :")
    print('   python -c "import tensorflow as tf; print(tf.__version__)"')
    print('   python -c "import tensorflow as tf; print(tf.config.list_physical_devices(\'GPU\'))"')
    print("\n📚 Ressources :")
    print("   Documentation GPU   : https://www.tensorflow.org/guide/gpu")
    print("   Guide compatibilité : https://www.tensorflow.org/install/source#gpu")


def print_failure_summary():
    print_header("INSTALLATION INTERROMPUE")
    print("❌ L'installation n'a pas pu se terminer.\n")
    print("🔧 Pistes de résolution :")
    print("   • Vérifiez les prérequis système (drivers, Python, conda…).")
    print("   • Consultez le guide manuel : tensorflow-gpu-setup.md")
    print("   • Relancez le script après avoir corrigé les problèmes signalés.")


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main() -> bool:
    print_header("Installation automatique TensorFlow GPU — Juin 2025")

    # 1. Détection du système
    print_step(1, "Détection du système")
    info = detect_system()
    print(f"   Système d'exploitation : {info['os']}")
    print(f"   Architecture           : {info['arch']}")
    print(f"   Version Python         : {info['py_version']}")
    print(f"   WSL détecté            : {info['is_wsl']}")
    print(f"   Conda disponible       : {info['has_conda']}")
    print(f"   uv disponible          : {info['has_uv']}")
    print(f"   GPU NVIDIA détecté     : {info['has_nvidia']}")
    print(f"   Environnement actif    : {info['active_env'] or 'aucun'}")

    # 2. Résolution de la plateforme
    print_step(2, "Résolution de la plateforme cible")
    platform_key = resolve_platform(info)
    if platform_key is None:
        print_failure_summary()
        return False

    config = PLATFORM_CONFIGS[platform_key]
    print(f"\n   Plateforme identifiée : {config['label']}")
    print(f"   Package TensorFlow    : {config['tf_package']}")
    print(f"   Python cible          : {config['python']}")

    # 3. Vérifications préalables
    print_step(3, "Vérification des prérequis")
    if not check_prerequisites(info, config):
        print_failure_summary()
        return False
    print("   ✅ Tous les prérequis sont satisfaits.")

    # 4. Création de l'environnement virtuel
    print_step(4, "Préparation de l'environnement virtuel")
    env_ready = create_env(info, config)
    if not env_ready:
        # create_env a déjà affiché la marche à suivre
        return False

    # 5. Installation des packages
    print_step(5, "Installation de TensorFlow")
    if not install_packages(info, config):
        print_failure_summary()
        return False

    print_success_summary()
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Installation interrompue par l'utilisateur.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Erreur inattendue : {e}")
        sys.exit(1)
