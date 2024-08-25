const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

// Define your URL list (you may generate or fetch this list dynamically)
const urls = [
    'https://www.telekom.de/hilfe/page1',
    'https://www.telekom.de/hilfe/page2',
    // Add more URLs here
];

const OUTPUT_FOLDER = './qa_output';
const MAX_CONCURRENT_REQUESTS = 5;

/**
 * Sanitize a URL to create a valid filename.
 */
const sanitizeFilename = (url) => {
    const filename = url.replace(/^https?:\/\//, '').replace(/[^a-zA-Z0-9]/g, '_');
    return filename;
};

/**
 * Extract and analyze the text content from the page.
 */
const extractAndAnalyzeContent = async (page) => {
    return await page.evaluate(() => {
        const content = [];
        
        // Collect text from potential question and answer containers
        const elements = document.querySelectorAll('h2, h3, p, div');
        elements.forEach(element => {
            const text = element.innerText.trim();
            if (text) {
                content.push(text);
            }
        });

        return content;
    });
};

/**
 * Determine if content has question-answer format.
 */
const containsQuestionAnswer = (content) => {
    // Simple heuristic to identify question-answer pairs
    const questions = content.filter(text => text.includes('?'));
    return questions.length > 0;
};

/**
 * Fetch and process content from a given URL.
 */
const fetchAndProcessPage = async (url) => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    let content = '';
    try {
        await page.goto(url, { waitUntil: 'networkidle2' });
        const pageContent = await extractAndAnalyzeContent(page);

        if (containsQuestionAnswer(pageContent)) {
            content += `URL: ${url}\n\n`;
            pageContent.forEach(text => content += `${text}\n\n`);
        }

        return content;
    } catch (error) {
        console.error(`Error processing ${url}: ${error.message}`);
    } finally {
        await browser.close();
    }
};

/**
 * Save content to a file.
 */
const saveContentToFile = async (filePath, content) => {
    try {
        await fs.promises.writeFile(filePath, content);
    } catch (error) {
        console.error(`Error writing to file ${filePath}: ${error.message}`);
    }
};

/**
 * Scrape a single URL and save its content.
 */
const scrapeUrl = async (url) => {
    const sanitizedFilename = `${sanitizeFilename(url)}.txt`;
    const filePath = path.join(OUTPUT_FOLDER, sanitizedFilename);

    const content = await fetchAndProcessPage(url);
    if (content) {
        await saveContentToFile(filePath, content);
    } else {
        console.log(`No relevant content found for ${url}. Skipping.`);
    }
};

/**
 * Process URLs in batches with concurrency limits.
 */
const processUrls = async () => {
    if (!fs.existsSync(OUTPUT_FOLDER)) {
        fs.mkdirSync(OUTPUT_FOLDER, { recursive: true });
    }

    for (let i = 0; i < urls.length; i += MAX_CONCURRENT_REQUESTS) {
        const batch = urls.slice(i, i + MAX_CONCURRENT_REQUESTS);
        await Promise.all(batch.map(url => scrapeUrl(url)));
    }

    console.log('All URLs processed successfully.');
};

// Start processing the URLs
processUrls();
