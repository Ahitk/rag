const axios = require('axios');
const fs = require('fs');
const xml2js = require('xml2js');

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

const sitemapUrl = 'https://www.telekom.de/ueber-das-unternehmen/robots/sitemap';
fetchSitemap(sitemapUrl);
