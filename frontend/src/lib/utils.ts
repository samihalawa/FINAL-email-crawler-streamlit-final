import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export type SiteConfig = {
  name: string
  description: string
  url: string
  links: {
    github: string
  }
}

export const siteConfig: SiteConfig = {
  name: "Autoclient.ai",
  description: "Lead Generation AI App",
  url: "https://autoclient.ai",
  links: {
    github: "https://github.com/autoclient/autoclient"
  }
} 