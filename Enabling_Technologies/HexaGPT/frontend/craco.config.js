module.exports = {
    webpack: {
        configure: (webpackConfig) => {
            // Find the rule that contains a 'oneOf' array
            const oneOfRule = webpackConfig.module.rules.find(rule => rule.oneOf);

            if (oneOfRule) {
                // Iterate over all the contained rules
                oneOfRule.oneOf.forEach(rule => {
                    // Check if 'source-map-loader' is used within the current rule
                    if (rule.loader && rule.loader.includes('source-map-loader')) {
                        // Apply the exclusion to this rule
                        if (!rule.exclude) {
                            rule.exclude = [];
                        }
                        rule.exclude.push(/node_modules\/@microsoft\/fetch-event-source/);
                    }
                });
            }

            return webpackConfig;
        },
    },
};
