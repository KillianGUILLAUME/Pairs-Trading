import asyncio
import os
import sys
from loguru import logger

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from live_platform.orchestrator import LiveOrchestrator

async def test_dry_run():
    logger.info("🧪 TEST DRY RUN : Plateforme Institutionnelle Live")
    
    # On initialise en mode Paper Trading pour la sécurité totale
    orchestrator = LiveOrchestrator(paper_trading=True)
    
    # On exécute un SEUL cycle pour valider la pipeline
    try:
        await orchestrator.run_cycle()
        logger.success("✅ Cycle de test terminé avec succès.")
    except Exception as e:
        logger.error(f"❌ Échec du cycle de test : {e}")
    finally:
        await orchestrator.bridge.close()

if __name__ == "__main__":
    asyncio.run(test_dry_run())
