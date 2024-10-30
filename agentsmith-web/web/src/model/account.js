import {logger} from "../utils/logging.js";

export class Account {
  constructor() {
    this.level = import.meta.env.VITE_APP_LOGGING_LEVEL || 'INFO';
  }
}


export class AccountStore {

  constructor(auth0) {
    this.baseUrl = import.meta.env.VITE_APP_DH_API_BASE_URL || "";
    this.auth0 = auth0;
    logger.info("Loading account store.", this.baseUrl)
  }

  accessToken = async () => {
    const hostname = window.location.hostname;
    const domain = ["localhost", "app.local"].includes(hostname) ? `${hostname}:${window.location.port}` : hostname;
    const protocol = hostname === "localhost" ? "http" : "https";

    const accessToken = hostname === "localhost" ?
        await this.auth0.getAccessTokenWithPopup({
          authorizationParams: {
            audience: `${protocol}://${domain}/api/`,
            scope: "read:current_user",
            redirect_uri: `${protocol}://${domain}`,
          },
        })
        : await this.auth0.getAccessTokenSilently({
          authorizationParams: {
            audience: `${protocol}://${domain}/api/`,
            scope: "read:current_user",
          },
        });

    const baseUrl = `${protocol}://${domain}/api`

    return {
      accessToken,
      baseUrl,
    };
  }

  postJSON = async (url, accessToken, data) => {

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${accessToken}`,
      },
      body: JSON.stringify(data),
    });

    const result = {
      status: response.status,
      data: await response.json(),
    }

    logger.debug("Success:", result);
    return result;
  }

  createProfile = async (form) => {
    logger.debug("Creating profile");
    const details = await this.accessToken()
    const url = `${details.baseUrl}/profiles/me`;
    return await this.postJSON(url, details.accessToken, form);
  }

  loadProfile = async () => {

    const details = await this.accessToken()

    const url = `${details.baseUrl}/profiles/me`;

    const response = await fetch(url, {
      headers: {
        Authorization: `Bearer ${details.accessToken}`,
      },
    });

    const result = {
      status: response.status,
      data: await response.json(),
    }

    return result;
  }
}


