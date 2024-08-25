const axios = require('axios');
const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');
const xml2js = require('xml2js');

// URL'leri alıp JSON dosyasına kaydetme
async function fetchSitemap(sitemapUrl) {
    try {
        const response = await axios.get(sitemapUrl);
        const result = await xml2js.parseStringPromise(response.data);
        const urls = result.urlset.url.map(url => url.loc[0]);

        fs.writeFileSync('sitemap.json', JSON.stringify(urls, null, 2));
        console.log('URLs have been saved to sitemap.json');
    } catch (error) {
        console.error('Error fetching or parsing the sitemap:', error.message);
    }
}

// Sitemap URL'si
const sitemapUrl = 'https://www.telekom.de/ueber-das-unternehmen/robots/sitemap';

// URL'leri işle ve txt dosyasına kaydet
const OUTPUT_FOLDER = './qa_output';
const MAX_CONCURRENT_REQUESTS = 5;

const sanitizeFilename = (url) => {
    return url.replace(/^https?:\/\//, '').replace(/[^a-zA-Z0-9]/g, '_') + '.txt';
};

const extractAndAnalyzeContent = async (page) => {
    return await page.evaluate(() => {
        const content = {
            question: '',
            answer: '',
            tips: [],
            links: []
        };

        const questionElement = document.querySelector('h2, h3, h4, h5, h6');
        if (questionElement) {
            content.question = questionElement.innerText.trim();
        }

        const answerElements = document.querySelectorAll('.article-text p, .article-text ul, .article-text ol');
        if (answerElements.length === 0) {
            console.log('Cevap elementleri bulunamadı.');
        }

        answerElements.forEach(element => {
            const text = element.innerText.trim();
            if (text) {
                const links = Array.from(element.querySelectorAll('a')).map(a => a.href);
                if (links.length > 0) {
                    content.links.push(...links);
                }

                if (text.startsWith('Unser Tipp') || text.startsWith('Hinweis')) {
                    content.tips.push(text);
                } else {
                    content.answer += `${text}\n`;
                }
            }
        });

        return content;
    });
};

const fetchAndProcessPage = async (url) => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    let content = '';
    try {
        await page.goto(url, { waitUntil: 'networkidle2' });

        try {
            await page.waitForSelector('body', { timeout: 10000 });
        } catch (error) {
            console.error(`Error waiting for selector on ${url}: ${error.message}`);
        }

        const pageContent = await extractAndAnalyzeContent(page);

        if (pageContent.question && (pageContent.answer || pageContent.tips.length > 0)) {
            content += `URL: ${url}\n\n`;
            content += `Soru: ${pageContent.question}\n\n`;
            
            if (pageContent.answer) {
                content += `Cevap: ${pageContent.answer.trim()}\n`;
            }

            if (pageContent.tips.length > 0) {
                pageContent.tips.forEach(tip => content += `Öneri: ${tip}\n`);
            }

            if (pageContent.links.length > 0) {
                content += `İç Linkler: ${pageContent.links.join(', ')}\n`;
            }
        } else {
            console.log(`İçerik bulunamadı: ${url}`);
        }

        return content;
    } catch (error) {
        console.error(`Error processing ${url}: ${error.message}`);
    } finally {
        await browser.close();
    }
};

const saveContentToFile = async (filePath, content) => {
    try {
        await fs.promises.writeFile(filePath, content);
    } catch (error) {
        console.error(`Error writing to file ${filePath}: ${error.message}`);
    }
};

const scrapeUrl = async (url) => {
    const sanitizedFilename = sanitizeFilename(url);
    const filePath = path.join(OUTPUT_FOLDER, sanitizedFilename);

    const content = await fetchAndProcessPage(url);
    if (content) {
        await saveContentToFile(filePath, content);
    } else {
        console.log(`No relevant content found for ${url}. Skipping.`);
    }
};

const processUrls = async () => {
    if (!fs.existsSync(OUTPUT_FOLDER)) {
        fs.mkdirSync(OUTPUT_FOLDER, { recursive: true });
    }

    const urls = JSON.parse(fs.readFileSync('sitemap.json', 'utf-8'));

    for (let i = 0; i < urls.length; i += MAX_CONCURRENT_REQUESTS) {
        const batch = urls.slice(i, i + MAX_CONCURRENT_REQUESTS);
        await Promise.all(batch.map(url => scrapeUrl(url)));
    }

    console.log('All URLs processed successfully.');
};

// Fetch sitemap and start processing
fetchSitemap(sitemapUrl).then(() => {
    processUrls();
});
