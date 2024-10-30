import './index.css'
import {createTheme,} from "@mui/material";


export const theme = createTheme({
    palette: {
        primary: {
            main: '#C85C2C', // Rust red (inspired by African clay)
            light: '#E47B4D',
            dark: '#A23D11',
            contrastText: '#FFFFFF',
        },
        secondary: {
            main: '#E8B341', // Golden yellow (inspired by African gold)
            light: '#FFCD66',
            dark: '#BE8A1D',
            contrastText: '#000000',
        },
        error: {
            main: '#D32F2F',
            light: '#EF5350',
            dark: '#C62828',
        },
        warning: {
            main: '#F4511E', // Deep orange (inspired by sunset)
            light: '#FF7043',
            dark: '#E64A19',
        },
        info: {
            main: '#0288D1',
            light: '#03A9F4',
            dark: '#01579B',
        },
        success: {
            main: '#2E7D32', // Forest green (inspired by tropical foliage)
            light: '#4CAF50',
            dark: '#1B5E20',
        },
        background: {
            default: '#FDF7E7', // Light sand color
            paper: '#FFFFFF',
        },
        text: {
            primary: '#2C1810', // Deep brown
            secondary: '#594A42', // Lighter brown
            disabled: 'rgba(0, 0, 0, 0.38)',
        },
        // Additional custom colors
        custom: {
            kente: {
                yellow: '#FFB300', // Kente cloth yellow
                green: '#357A38', // Kente cloth green
                red: '#C62828', // Kente cloth red
                black: '#212121', // Kente cloth black
            },
            earth: {
                clay: '#A65D57', // Terra cotta
                sand: '#DBC1AC', // Desert sand
                bark: '#4E342E', // Tree bark brown
                stone: '#757575', // Stone gray
            },
            nature: {
                savanna: '#8D6E63', // Savanna brown
                forest: '#2E7D32', // Forest green
                sunset: '#FF7043', // Sunset orange
                sky: '#0288D1', // Sky blue
            }
        }
    },
    // Optional: Adding typography settings to complement the African theme
    typography: {
        fontFamily: '"Roboto", "Arial", sans-serif',
        h1: {
            color: '#2C1810',
            fontWeight: 700,
        },
        h2: {
            color: '#2C1810',
            fontWeight: 600,
        },
        body1: {
            color: '#2C1810',
        },
    },
});
