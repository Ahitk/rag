const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

const INPUT_FILE = './scrape_telekom/sitemap.json';
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
        // Example of more specific selectors; adjust as needed
        const questionElements = document.querySelectorAll('.question-class'); // Adjust to actual class/ID
        const answerElements = document.querySelectorAll('.answer-class'); // Adjust to actual class/ID

        questionElements.forEach(element => {
            const text = element.innerText.trim();
            if (text) {
                content.push(`Q: ${text}`);
            }
        });

        answerElements.forEach(element => {
            const text = element.innerText.trim();
            if (text) {
                content.push(`A: ${text}`);
            }
        });

        return content;
    });
};

/**
 * Determine if content has question-answer format.
 */
const containsQuestionAnswer = (content) => {
    const questions = content.filter(text => text.startsWith('Q:'));
    return questions.length > 0;
};

/**
 * Fetch and process content from a given URL.
 */
const fetchAndProcessPage = async (url) => {
    const browser = await puppeteer.launch({ headless: true });
    const page = await browser.newPage();
    let content = '';
    try {
        await page.goto(url, { waitUntil: 'networkidle2' });
        const pageContent = await extractAndAnalyzeContent(page);

        // Log content for debugging
        console.log(`Content extracted from ${url}:`, pageContent);

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
        console.log(`File saved: ${filePath}`); // Log successful file saving
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
 * Read URLs from the JSON file.
 */
const readUrlsFromFile = async (filePath) => {
    try {
        const data = await fs.promises.readFile(filePath, 'utf-8');
        return JSON.parse(data);
    } catch (error) {
        console.error(`Error reading file ${filePath}: ${error.message}`);
        return [];
    }
};

/**
 * Process URLs in batches with concurrency limits.
 */
const processUrls = async () => {
    if (!fs.existsSync(OUTPUT_FOLDER)) {
        fs.mkdirSync(OUTPUT_FOLDER, { recursive: true });
        console.log(`Output folder created: ${OUTPUT_FOLDER}`); // Log folder creation
    }

    const urls = await readUrlsFromFile(INPUT_FILE);
    if (urls.length === 0) {
        console.log('No URLs to process.');
        return;
    }

    console.log(`Processing ${urls.length} URLs...`);

    for (let i = 0; i < urls.length; i += MAX_CONCURRENT_REQUESTS) {
        const batch = urls.slice(i, i + MAX_CONCURRENT_REQUESTS);
        await Promise.all(batch.map(url => scrapeUrl(url)));
        console.log(`Processed batch ${Math.floor(i / MAX_CONCURRENT_REQUESTS) + 1}`);
    }

    console.log('All URLs processed successfully.');
};

// Start processing the URLs
processUrls();
