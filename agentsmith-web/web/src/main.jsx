import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import {Auth0Provider} from "@auth0/auth0-react";
import { BrowserRouter, useNavigate } from 'react-router-dom';
import {ThemeProvider} from "@mui/material";
import {IntlProvider} from "react-intl";
import {theme} from "./theme.jsx";

const Auth0ProviderWithRedirectCallback = ({
  // eslint-disable-next-line react/prop-types
  children,
  ...props
}) => {
  const navigate = useNavigate();

  const onRedirectCallback = (appState) => {
    navigate((appState && appState.returnTo) || window.location.pathname);
  };

  return (
      <Auth0Provider onRedirectCallback={onRedirectCallback} {...props}>
        {children}
      </Auth0Provider>
  );
};


ReactDOM.createRoot(document.getElementById('root')).render(
    <BrowserRouter>
      <Auth0ProviderWithRedirectCallback
          domain={import.meta.env.VITE_AUTH0_DOMAIN}
          clientId={import.meta.env.VITE_AUTH0_CLIENT_ID}
          authorizationParams={{
            audience: import.meta.env.VITE_APP_AUDIENCE,
            redirect_uri: window.location.origin,
          }} >
        <ThemeProvider theme={theme}>
          <IntlProvider locale={"en"}>
            <App />
          </IntlProvider>
        </ThemeProvider>
      </Auth0ProviderWithRedirectCallback>
    </BrowserRouter>,
)
