module.exports = {
    module: {
        rules: [
            {
                test: /\.js$/,
                use: ['source-map-loader'],
                enforce: 'pre',
                exclude: /node_modules\/@microsoft\/fetch-event-source/, // Correctly exclude source maps for this package
            },
            // other rules...
        ],
    },
};
