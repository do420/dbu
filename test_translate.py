import asyncio
from agents import create_agent

async def test():
    # Test İngilizce'den Türkçe'ye çeviri
    config = {'target_language': 'tr'}
    agent = create_agent('google_translate', config, 'Translate text accurately')
    result = await agent.process('Hello world, this is a test of the Google Translate agent.')
    print(f"İngilizce -> Türkçe: {result['output']}")
    
    # Test Türkçe'den İngilizce'ye çeviri
    config = {'target_language': 'en'}
    agent = create_agent('google_translate', config, 'Translate text accurately')
    result = await agent.process('Merhaba dünya, bu Google Translate ajanının bir testidir.')
    print(f"Türkçe -> İngilizce: {result['output']}")
    
    return "Test tamamlandı"

if __name__ == "__main__":
    result = asyncio.run(test())
    print(result)